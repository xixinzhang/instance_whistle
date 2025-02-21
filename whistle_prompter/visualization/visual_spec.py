from .base import *

import numpy as np

def get_axis_ticks(shape, pix_tick=False):
    """Generate tick marks and labels for spectrogram visualization"""
    if not pix_tick:
        y_ticks = np.linspace(0, shape[0] - 1, num=10, endpoint=True, dtype=int)
        y_labels = np.linspace(5, 50, num=10, endpoint=True, dtype=int)[::-1]
        x_ticks = np.linspace(0, shape[1] - 1, num=20, endpoint=True, dtype=int)
        x_labels = np.round(np.linspace(0, 3, num=20, endpoint=True), 2)
    else:
        y_ticks = np.linspace(0, shape[0] - 1, num=10, endpoint=True, dtype=int)
        y_labels = y_ticks[::-1]
        x_ticks = np.linspace(0, shape[1] - 1, num=20, endpoint=True, dtype=int)
        x_labels = x_ticks
        
    return x_ticks, x_labels, y_ticks, y_labels

def get_margins(axis=False):
    """Get margin settings based on whether axis is shown"""
    if axis:
        return {
            'left_margin_px': 60,
            'right_margin_px': 20,
            'top_margin_px': 10,
            'bottom_margin_px': 60
        }
    return {
        'left_margin_px': 0,
        'right_margin_px': 0,
        'top_margin_px': 0,
        'bottom_margin_px': 0
    }

def plot_spec(spec, filename=None, save_dir="outputs", cmap='bone', 
               pix_tick=False, axis=False, dpi=300):
    """Plot spectrogram"""
    x_ticks, x_labels, y_ticks, y_labels = get_axis_ticks(spec.shape, pix_tick)
    
    options = {
        'cmap': cmap,
        'x_ticks_labels': [x_ticks, x_labels],
        'y_ticks_labels': [y_ticks, y_labels],
        'dpi': dpi,
        **get_margins(axis)
    }
    
    if axis:
        options.update({
            'x_label': 'Time (s)',
            'y_label': 'Frequency (kHz)'
        })
    
    visualize_array(
        spec,
        filename=filename,
        save_dir=save_dir,
        options=options
    )

def plot_binary_mask(mask, filename=None, save_dir="outputs", 
                     pix_tick=False, axis=False, dpi=300):
    """Plot binary mask"""
    mask = mask.squeeze()
    x_ticks, x_labels, y_ticks, y_labels = get_axis_ticks(mask.shape, pix_tick)
    
    options = {
        'cmap': 'bone',
        'x_ticks_labels': [x_ticks, x_labels],
        'y_ticks_labels': [y_ticks, y_labels],
        'dpi': dpi,
        **get_margins(axis)
    }
    
    if axis:
        options.update({
            'x_label': 'Time (s)',
            'y_label': 'Frequency (kHz)'
        })
    
    visualize_array(
        mask,
        filename=filename,
        save_dir=save_dir,
        options=options
    )

def plot_mask_over_spec(spec, mask, filename=None, save_dir="outputs", 
                        cmap='bone', random_colors='none', pix_tick=False, 
                        axis=False, dpi=300):
    """Plot mask overlay on spectrogram"""
    x_ticks, x_labels, y_ticks, y_labels = get_axis_ticks(spec.shape, pix_tick)
    
    options = {
        'cmap': cmap,
        'x_ticks_labels': [x_ticks, x_labels],
        'y_ticks_labels': [y_ticks, y_labels],
        'dpi': dpi,
        **get_margins(axis)
    }
    
    if axis:
        options.update({
            'x_label': 'Time (s)',
            'y_label': 'Frequency (kHz)'
        })
    
    overlays = [
        MaskOverlay(
            mask=mask,
            random_colors=random_colors,
            alpha=1.0
        )
    ]
    
    visualize_array(
        spec,
        overlays=overlays,
        filename=filename,
        save_dir=save_dir,
        options=options
    )

def plot_points_over_spec(spec, points, filename=None, save_dir="outputs", 
                          cmap='bone', pix_tick=False, axis=False, dpi=300):
    """Plot points overlay on spectrogram"""
    x_ticks, x_labels, y_ticks, y_labels = get_axis_ticks(spec.shape, pix_tick)
    
    options = {
        'cmap': cmap,
        'x_ticks_labels': [x_ticks, x_labels],
        'y_ticks_labels': [y_ticks, y_labels],
        'dpi': dpi,
        **get_margins(axis)
    }
    
    if axis:
        options.update({
            'x_label': 'Time (s)',
            'y_label': 'Frequency (kHz)'
        })
    
    overlays = [
        PointOverlay(
            points=points,
            color="green",
            markersize=1
        )
    ]
    
    visualize_array(
        spec,
        overlays=overlays,
        filename=filename,
        save_dir=save_dir,
        options=options
    )

