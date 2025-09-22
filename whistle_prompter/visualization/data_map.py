import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import COASTLINE, LAND, OCEAN, BORDERS, STATES
import numpy as np

# Data from the provided image
species_data = {
    'Common dolphin': {'loc': 'Southern California Bight', 'lat': 33.5, 'lon': -118.5, 'refined_whistles': 8092, 'color': '#1f77b4'},
    'Bottlenose dolphin': {'loc': 'Southern California Bight', 'lat': 33.5, 'lon': -118.5, 'refined_whistles': 3505, 'color': '#ff7f0e'},
    'Melon-headed whale': {'loc': 'Palmyra Atoll', 'lat': 5.8, 'lon': -162.1, 'refined_whistles': 5424, 'color': '#2ca02c'},
    'Spinner dolphin': {'loc': 'Palmyra Atoll', 'lat': 5.8, 'lon': -162.1, 'refined_whistles': 3031, 'color': '#d62728'},
}

# Revised marine biology landmark
landmarks = {
    'Hawaii': {'lat': 21.5, 'lon': -158.0}
}

# Use refined_whistles for bubble size
all_whistles = [data['refined_whistles'] for data in species_data.values()]
max_whistles = max(all_whistles)
min_whistles = min(all_whistles)
size_min = 100
size_max = 2000

# Create the figure and a Plate Carree projection
fig, ax = plt.subplots(1, 1, figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# Set map extent
ax.set_extent([-180, -110, 0, 40], crs=ccrs.PlateCarree())

# Add geographical features
ax.add_feature(LAND, facecolor='lightgray', zorder=1)
ax.add_feature(OCEAN, facecolor='azure', zorder=1)
ax.add_feature(COASTLINE, zorder=2)
ax.add_feature(BORDERS, linestyle=':', zorder=2)
ax.add_feature(STATES, zorder=2)
ax.gridlines(draw_labels=True, xlocs=np.arange(-180, -109, 20), ylocs=np.arange(0, 41, 10), linestyle='--')

# Location offsets
offsets = {
    'Southern California Bight': [(0.5, -0.5), (-0.5, 0.5)],
    'Palmyra Atoll': [(0.5, -0.5), (-0.5, 0.5)]
}
loc_idx = {'Southern California Bight': 0, 'Palmyra Atoll': 0}

# Plot each species as a bubble
for species, data in species_data.items():
    loc = data['loc']
    idx = loc_idx[loc]
    offset_lon, offset_lat = offsets[loc][idx]

    bubble_size = size_min + (data['refined_whistles'] - min_whistles) / (max_whistles - min_whistles) * (size_max - size_min)

    ax.scatter(data['lon'] + offset_lon, data['lat'] + offset_lat,
               s=bubble_size,
               color=data['color'],
               alpha=0.7,
               edgecolors='black',
               linewidths=1,
               transform=ccrs.PlateCarree(),
               zorder=3)

    loc_idx[loc] += 1

# Add main location markers and labels
ax.scatter([-118.5, -162.1], [33.5, 5.8], color='black', marker='x', s=100, zorder=4)
ax.text(-118.5, 33.5 - 2, 'Southern California Bight', horizontalalignment='center', fontsize=12, style='italic', color='dimgray', transform=ccrs.PlateCarree(), zorder=4)
ax.text(-162.1, 5.8 + 2, 'Palmyra Atoll', horizontalalignment='center', fontsize=12, style='italic', color='dimgray', transform=ccrs.PlateCarree(), zorder=4)

# Add landmark labels without markers
for landmark, coords in landmarks.items():
    ax.text(coords['lon'], coords['lat'] + 2, landmark, horizontalalignment='center', fontsize=10, style='italic', color='dimgray', transform=ccrs.PlateCarree(), zorder=4)

# Create a custom legend for bubble sizes
legend_whistles = [3000, 5000, 8000]
legend_sizes = [size_min + (w - min_whistles) / (max_whistles - min_whistles) * (size_max - size_min) for w in legend_whistles]
legend_handles_size = []
for size, label in zip(legend_sizes, legend_whistles):
    legend_handles_size.append(ax.scatter([], [], s=size, c='gray', alpha=0.7, edgecolors='black', label=f'{label} whistles'))

# Create a separate legend for species color
legend_handles_color = [plt.Rectangle((0,0),1,1, color=species_data[s]['color'], ec="k") for s in species_data]
legend_labels_color = list(species_data.keys())

# Place the legends
first_legend = ax.legend(legend_handles_color, legend_labels_color, title='Species', loc='upper left', frameon=True, edgecolor='black', fancybox=False, facecolor='white')
ax.add_artist(first_legend)
ax.legend(handles=legend_handles_size, labels=[f'{w} whistles' for w in legend_whistles], title='# of Refined Whistles', loc='lower right', frameon=True, edgecolor='black', fancybox=False, facecolor='white')

# Add the 'North Pacific Ocean' label with a higher zorder
ax.text(-165, 37, 'North Pacific Ocean', horizontalalignment='left', fontsize=14, color='darkslategray', transform=ccrs.PlateCarree(), style='italic', fontweight='bold', alpha=0.5, zorder=4)

# Finalize the plot
# ax.set_title('Refined Whistle Count by Species and Location', fontsize=16)
plt.tight_layout()
plt.savefig('whistle_count_map.png', dpi=300, bbox_inches='tight')
plt.show()