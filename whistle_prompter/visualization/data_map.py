import matplotlib.pyplot as plt
import netCDF4
import cartopy.crs as ccrs
from cartopy.feature import COASTLINE, LAND, OCEAN, BORDERS, STATES
import numpy as np

# Data from the provided image
species_data = [
    {'species': 'Common Dolphin', 'loc': 'Southern California Bight', 'lat': 33.5, 'lon': -118.5, 'refined_whistles': 8092, 'color': '#ffd600'},      # Yellow
    {'species': 'Bottlenose Dolphin', 'loc': 'Southern California Bight', 'lat': 33.5, 'lon': -118.5, 'refined_whistles': 2253, 'color': '#e53935'}, # Red
    {'species': 'Bottlenose Dolphin', 'loc': 'Palmyra Atoll', 'lat': 5.8, 'lon': -162.1, 'refined_whistles': 1252, 'color': '#e53935'},               # Red
    {'species': 'Melon-headed Whale', 'loc': 'Palmyra Atoll', 'lat': 5.8, 'lon': -162.1, 'refined_whistles': 5424, 'color': '#00bcd4'},               # Cyan
    {'species': 'Spinner Dolphin', 'loc': 'Palmyra Atoll', 'lat': 5.8, 'lon': -162.1, 'refined_whistles': 3031, 'color': '#ff9800'},                  # Orange
]

# Revised marine biology landmark
landmarks = {
    'Hawaii': {'lat': 21.5, 'lon': -158.0}
}

# Use refined_whistles for bubble size
all_whistles = [data['refined_whistles'] for data in species_data]
max_whistles = max(all_whistles)
min_whistles = min(all_whistles)
size_min = 100
size_max = 2000

# Create the figure and a Plate Carree projection
fig, ax = plt.subplots(1, 1, figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# Set map extent
ax.set_extent([-180, -110, 0, 40], crs=ccrs.PlateCarree())

# Add geographical features

# Load ETOPO1 bathymetry data (make sure ETOPO1_Bed_g_gmt4.grd is in your working directory)
try:
    nc = netCDF4.Dataset('ETOPO1_Bed_g_gmt4.grd')
    lons = nc.variables['x'][:]
    lats = nc.variables['y'][:]
    elev = nc.variables['z'][:]
    # Subset to map extent for speed and memory
    lon_mask = (lons >= -180) & (lons <= -110)
    lat_mask = (lats >= 0) & (lats <= 40)
    lons_sub = lons[lon_mask]
    lats_sub = lats[lat_mask]
    elev_sub = elev[np.ix_(lat_mask, lon_mask)]
    import matplotlib.patheffects as path_effects
    contour = ax.contourf(lons_sub, lats_sub, elev_sub, levels=np.arange(-6000, 0, 500), cmap='Blues_r', transform=ccrs.PlateCarree(), zorder=0)
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', label='Ocean Depth (m)', fraction=0.027, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
except Exception as e:
    print(f"Bathymetry data not loaded: {e}")

ax.add_feature(LAND, facecolor='lightgray', zorder=1)
ax.add_feature(COASTLINE, zorder=2)
ax.add_feature(BORDERS, linestyle=':', zorder=2)
ax.add_feature(STATES, zorder=2)
ax.gridlines(draw_labels=True, xlocs=np.arange(-180, -109, 20), ylocs=np.arange(0, 41, 10), linestyle='--')

# Location offsets
offsets = {
    'Southern California Bight': [(-0.5, -3.5), (-3, 0.5)],  # move circles further away
    'Palmyra Atoll': [(2.8, -0.5), (2, 3.5), (2.5, 1)]
}
loc_idx = {'Southern California Bight': 0, 'Palmyra Atoll': 0}

# Add main location markers and labels
ax.scatter([-118.5, -162.1], [33.5, 5.8], color='black', marker='x', s=150, zorder=4)
scb_text = ax.text(-128 - 3, 33 - 1.5, 'Southern California Bight', horizontalalignment='center', fontsize=22, style='italic', color='white', transform=ccrs.PlateCarree(), zorder=2)
scb_text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
pa_text = ax.text(-165, 3, 'Palmyra Atoll', horizontalalignment='center', fontsize=22, style='italic', color='white', transform=ccrs.PlateCarree(), zorder=4)
pa_text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])


# Plot each species as a bubble
# Plot each species as a bubble
for data in species_data:
    species = data['species']
    loc = data['loc']
    idx = loc_idx[loc]
    offset_lon, offset_lat = offsets[loc][idx]

    # New circle position
    circle_lon = data['lon'] + offset_lon
    circle_lat = data['lat'] + offset_lat
    cross_lon = data['lon']
    cross_lat = data['lat']

    bubble_size = size_min + (data['refined_whistles'] - min_whistles) / (max_whistles - min_whistles) * (size_max - size_min)
    # Calculate radius in degrees (approximate)
    # s in points^2, so radius in points: r = sqrt(s/pi)
    radius = np.sqrt(bubble_size/np.pi)/15

    # Direction vector from cross to circle
    dx = circle_lon - cross_lon
    dy = circle_lat - cross_lat
    dist = np.sqrt(dx**2 + dy**2)
    # The line should start at the cross and end at the circle's edge
    line_end_lon = cross_lon + dx * ((dist - radius) / dist)
    line_end_lat = cross_lat + dy * ((dist - radius) / dist)

    # Draw line from cross to circumference of circle
    ax.plot([cross_lon, line_end_lon], [cross_lat, line_end_lat], color="#FF5900", linewidth=1.5, linestyle='-', transform=ccrs.PlateCarree(), zorder=2)

    # Draw circle
    ax.scatter(circle_lon, circle_lat,
               s=bubble_size,
               color=data['color'],
               alpha=0.7,
               edgecolors='black',
               linewidths=1,
               transform=ccrs.PlateCarree(),
               zorder=3)

    loc_idx[loc] += 1


# Add landmark labels without markers
for landmark, coords in landmarks.items():
    lm_text = ax.text(coords['lon']+8, coords['lat']-2 , landmark, horizontalalignment='center', fontsize=22, style='italic', color='white', transform=ccrs.PlateCarree(), zorder=4)
    lm_text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

# Create a custom legend for bubble sizes
legend_whistles = [3000, 5000, 8000]
legend_sizes = [size_min + (w - min_whistles) / (max_whistles - min_whistles) * (size_max - size_min) for w in legend_whistles]
legend_handles_size = []
for size, label in zip(legend_sizes, legend_whistles):
    legend_handles_size.append(ax.scatter([], [], s=size, c='gray', alpha=0.7, edgecolors='black', label=f'{label} whistles'))

# Create a separate legend for species color
legend_color_map = {}
for entry in species_data:
    legend_color_map.setdefault(entry['species'], entry['color'])

legend_handles_color = [plt.Rectangle((0,0),1,1, color=color, ec="k") for color in legend_color_map.values()]
legend_labels_color = list(legend_color_map.keys())

# Place the legends
first_legend = ax.legend(
    legend_handles_color,
    legend_labels_color,
    title='Species',
    loc='upper left',
    frameon=True,
    edgecolor='black',
    fancybox=False,
    facecolor='white',
    fontsize=14,
    title_fontsize=16
)
ax.add_artist(first_legend)
ax.legend(
    handles=legend_handles_size,
    labels=[f'{w} whistles' for w in legend_whistles],
    title='Number of whistles',
    loc='lower right',
    frameon=True,
    edgecolor='black',
    fancybox=False,
    facecolor='white',
    fontsize=14,
    title_fontsize=16
)

# Add the 'North Pacific Ocean' label with a higher zorder
ocean_text = ax.text(-161, 36, 'North Pacific Ocean', horizontalalignment='left', fontsize=30, color='white', transform=ccrs.PlateCarree(), style='italic', fontweight='bold', alpha=0.9, zorder=4)
import matplotlib.patheffects as path_effects
ocean_text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

# Finalize the plot
# ax.set_title('Refined Whistle Count by Species and Location', fontsize=16)
plt.tight_layout()
plt.savefig('whistle_count_map.png', bbox_inches='tight')
plt.show()