import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

FLOAT_TYPE = [np.float32, np.float64, torch.float32, torch.float64]
INT_TYPE = [np.uint8, np.uint16, np.int32, np.int64, torch.int32, torch.int64]

class VisualizationOverlay:
    """Base class for visualization overlays"""
    def draw(self, ax):
        raise NotImplementedError

class PointOverlay(VisualizationOverlay):
    def __init__(self, points, color="green", marker="*", markersize=5):
        self.points = points
        self.color = color
        self.marker = marker
        self.markersize = markersize
    
    def draw(self, ax):
        if not self.points:
            return
        row, col = zip(*self.points)
        ax.scatter(col, row, color=self.color, marker=self.marker, s=self.markersize)

class BoxOverlay(VisualizationOverlay):
    def __init__(self, boxes, edgecolor="red", linewidth=2, facecolor=(0,0,0,0), fill=False, zorder=10):
        self.boxes = boxes
        self.edgecolor = edgecolor
        self.linewidth = linewidth
        self.facecolor = facecolor
        self.fill = fill
        self.zorder = zorder
    
    def draw(self, ax):
        if not self.boxes:
            return
        for box in self.boxes:
            x0, y0, w, h = box
            rect = Rectangle((x0, y0), w, h, 
                           edgecolor=self.edgecolor,
                           linewidth=self.linewidth,
                           facecolor=self.facecolor,
                           fill=self.fill,
                           zorder=self.zorder)
            ax.add_patch(rect)

class MaskOverlay(VisualizationOverlay):
    def __init__(self, mask, class_colors=None, mask_color=[255, 0, 102], 
                 random_colors='predefined', predefined_colors=None, alpha=0.6):
        """
        Args:
            mask: The segmentation mask
            class_colors: Dictionary mapping class ids to colors
            mask_color: Default color for masks [R,G,B] in range [0,255]
            random_colors: One of ['none', 'predefined', 'full']
                - 'none': Use mask_color for all classes
                - 'predefined': Randomly select from predefined_colors
                - 'full': Generate completely random colors
            predefined_colors: List of RGB tuples to use when random_colors='predefined'
            alpha: Transparency value between 0 and 1
        """
        self.mask = mask
        self.class_colors = class_colors or {}
        self.mask_color = np.array(mask_color) / 255
        self.random_colors = random_colors
        self.alpha = alpha
        
        self.predefined_colors = predefined_colors or [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (1.0, 0.0, 1.0),  # Magenta
            (1.0, 0.84, 0.0), # Gold
            (0.0, 1.0, 1.0),  # Cyan
            (1.0, 0.55, 0.0)  # Dark orange
        ]
    
    def draw(self, ax):
        if self.mask is None:
            return
            
        if self.mask.ndim == 3:
            self.mask = self.mask.squeeze()
            
        height, width = self.mask.shape
        colored_mask = np.zeros((height, width, 4), dtype=np.float32)
        
        unique_classes = np.unique(self.mask)
        for cls in unique_classes:
            if cls == 0:  # Skip background
                continue
                
            if cls in self.class_colors:
                color = self.class_colors[cls]
            elif self.random_colors == 'predefined':
                base_color = np.array(self.predefined_colors[
                    np.random.randint(0, len(self.predefined_colors))])
                color = np.concatenate([base_color, np.array([self.alpha])])
            elif self.random_colors == 'full':
                base_color = np.random.random(3)
                color = np.concatenate([base_color, np.array([self.alpha])])
            else:  # 'none'
                color = np.concatenate([self.mask_color, np.array([self.alpha])])
                
            colored_mask[self.mask == cls] = color.reshape(1, 1, 4)
            
        ax.imshow(colored_mask, alpha=self.alpha)

def preprocess_array(array):
    """Preprocess input array for visualization"""
    assert isinstance(array, (torch.Tensor, np.ndarray)), "Input must be a PyTorch tensor or NumPy array"
    
    array = array.squeeze()
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
        
    if array.ndim == 3 and array.shape[0] <= 4:
        array = np.moveaxis(array, 0, -1)  # Convert CHW -> HWC
        
    if array.dtype in FLOAT_TYPE and (np.max(array) > 1 or np.min(array) < 0):
        raise ValueError("Array of float values should be normalized to range [0, 1]")
        
    if array.ndim not in [2, 3]:
        raise ValueError("Input array must have 2 (grayscale) or 3 (HWC) dimensions")
        
    return array

def setup_figure(height, width, margins, dpi=100):
    """Setup matplotlib figure with proper dimensions and margins"""
    left_px, right_px, top_px, bottom_px = margins
    
    fig, ax = plt.subplots(
        figsize=(
            (width + left_px + right_px) / 100,
            (height + top_px + bottom_px) / 100
        ),
        dpi=dpi
    )
    
    # Calculate margin fractions
    total_width = width + left_px + right_px
    total_height = height + top_px + bottom_px
    margins = {
        'left': left_px / total_width,
        'right': 1 - (right_px / total_width),
        'top': 1 - (top_px / total_height),
        'bottom': bottom_px / total_height
    }
    
    plt.subplots_adjust(**margins)
    return fig, ax

def setup_axes(ax, array_shape, options):
    """Configure axis properties based on options"""
    height, width = array_shape[:2]
    
    if options.get('x_ticks_labels'):
        x_ticks = options['x_ticks_labels']
        if isinstance(x_ticks, list):
            ax.set_xticks(x_ticks[0], x_ticks[1], fontsize=options.get('tick_size', 12))
        elif isinstance(x_ticks, int):
            ticks = np.linspace(0, width, x_ticks, endpoint=True, dtype=int)
            ax.set_xticks(ticks, fontsize=options.get('tick_size', 12))
            
    if options.get('y_ticks_labels'):
        y_ticks = options['y_ticks_labels']
        if isinstance(y_ticks, list):
            ax.set_yticks(y_ticks[0], y_ticks[1], fontsize=options.get('tick_size', 12))
        elif isinstance(y_ticks, int):
            ticks = np.linspace(0, height, y_ticks, endpoint=True, dtype=int)
            ax.set_yticks(ticks, fontsize=options.get('tick_size', 12))
            
    if options.get('x_label'):
        ax.set_xlabel(options['x_label'], fontsize=options.get('label_size', 12))
    if options.get('y_label'):
        ax.set_ylabel(options['y_label'], fontsize=options.get('label_size', 12))
        
    if not any(options.get(m, 0) for m in ['left_margin_px', 'bottom_margin_px']):
        ax.axis('off')

def visualize_array(
    array,
    overlays=None,
    filename=None,
    save_dir="outputs",
    margins=(0, 0, 0, 0),  # left, right, top, bottom
    options=None,
    dpi=100,
):
    """
    Visualize an array with optional overlays and customization options.
    
    Args:
        array: torch.Tensor or np.ndarray to visualize
        overlays: list of VisualizationOverlay objects
        filename: optional filename to save the visualization
        save_dir: directory to save the visualization
        margins: tuple of (left, right, top, bottom) margins in pixels
        options: dict of visualization options including:
            - cmap: colormap for grayscale images
            - x_ticks_labels, y_ticks_labels: tick configurations
            - x_label, y_label: axis labels
            - tick_size, label_size: font sizes
        dpi: dots per inch for the output figure
    """
    array = preprocess_array(array)
    options = options or {}
    
    # Setup figure and axes
    fig, ax = setup_figure(array.shape[0], array.shape[1], margins, dpi)
    
    # Display the base array
    if array.ndim == 2:
        if array.dtype in FLOAT_TYPE:
            ax.imshow(array, cmap=options.get('cmap', 'viridis'), vmin=0, vmax=1)
        elif array.dtype in INT_TYPE:
            ax.imshow(array, cmap=options.get('cmap', 'viridis'), vmin=0, vmax=array.max())
        else:
            raise ValueError("Array dtype not supported")
    else:  # RGB or multi-channel
        ax.imshow(array)
    
    # Draw overlays
    if overlays:
        for overlay in overlays:
            overlay.draw(ax)
    
    # Setup axes properties
    setup_axes(ax, array.shape, options)
    
    # Save or display
    if filename and save_dir:
        save_path = f"{save_dir}/{filename}.png"
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
        plt.close()
    else:
        plt.show()