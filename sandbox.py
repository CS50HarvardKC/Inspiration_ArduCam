import smopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
from shapely.geometry import LineString, Point
import math

# Geographic boundaries
# Max: 27.37487861916057, -82.4524405368127
# Min: 27.37283293275336, -82.45435844706832
latlim = [27.37283293275336, 27.37487861916057]
lonlim = [-82.45435844706832, -82.4524405368127]

# Load map using smopy
zoom = 18
map_obj = smopy.Map((latlim[0], lonlim[0], latlim[1], lonlim[1]), z=zoom)

# Data storage
clicked_points = []
lines = []
line_objects = []
point_objects = []
layer_labels = []
visibility_flags = []

# Function to calculate distance using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Convert (x, y) screen coordinates to (latitude, longitude)
def xy_to_latlong(x, y, x1, y1, lat1, long1, x2, y2, lat2, long2):
    dx_pixels = x2 - x1
    dy_pixels = y2 - y1
    lat_per_pixel = (lat2 - lat1) / dy_pixels
    lon_per_pixel = (long2 - long1) / dx_pixels
    lat = lat1 + (y - y1) * lat_per_pixel
    lon = long1 + (x - x1) * lon_per_pixel
    return lat, lon

# Create figure
fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(map_obj.to_numpy(), extent=[0, map_obj.w, map_obj.h, 0])
ax.set_title("Interactive Map with Layer Control")

# Click event to add points
def onclick(event):
    if event.inaxes != ax:
        return
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        #print(x, y)
        # lat, lon = smopy.num2deg(x, y, zoom)
        lat, lon = xy_to_latlong(x, y, 
                                 496.835936, 328.017110, 27.375712, -82.452596, 
                                 496.014007, 385.004197, 27.375209, -82.452533)
        point = ax.scatter(x, y, c='red', s=50, marker='o', label='Point')
        print(lat, lon)
        point_objects.append(point)
        clicked_points.append((x, y))
        layer_labels.append(f"Point {len(point_objects)}")
        visibility_flags.append(True)
        update_table()
        fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)

# Function to connect points with a line
def connect_points(event):
    if len(clicked_points) >= 2:
        x_vals, y_vals = zip(*clicked_points)
        line, = ax.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'Line {len(line_objects) + 1}')
        lines.append(clicked_points.copy())
        line_objects.append(line)
        layer_labels.append(f"Line {len(line_objects)}")
        visibility_flags.append(True)
        clicked_points.clear()
        update_table()
        fig.canvas.draw()

# Function to toggle visibility
def toggle_visibility(label):
    index = layer_labels.index(label)
    visibility_flags[index] = not visibility_flags[index]

    if label.startswith("Point"):
        point_objects[index].set_visible(visibility_flags[index])
    elif label.startswith("Line"):
        line_objects[index - len(point_objects)].set_visible(visibility_flags[index])

    fig.canvas.draw()

# Function to remove selected (checked) points and lines
def remove_selected(event):
    global layer_labels, visibility_flags, point_objects, line_objects, lines

    indices_to_remove = [i for i, vis in enumerate(visibility_flags) if not vis]  # Remove unchecked items
    indices_to_remove.reverse()  # Reverse order to prevent shifting issues during deletion

    for index in indices_to_remove:
        if index < len(point_objects):
            point_objects[index].remove()
            del point_objects[index]
        else:
            adj_index = index - len(point_objects)
            line_objects[adj_index].remove()
            del line_objects[adj_index]
            del lines[adj_index]

        del layer_labels[index]
        del visibility_flags[index]

    update_table()
    fig.canvas.draw()

# Function to clear all points and lines
def clear_all(event):
    global layer_labels, visibility_flags, point_objects, line_objects, lines, clicked_points, checkbuttons, table_ax, check_ax, remove_button_ax

    # Remove all plotted elements
    for point in point_objects:
        point.remove()
    for line in line_objects:
        line.remove()

    # Reset all lists
    point_objects.clear()
    line_objects.clear()
    lines.clear()
    clicked_points.clear()
    layer_labels.clear()
    visibility_flags.clear()

    # Properly remove UI elements
    if "table_ax" in globals() and table_ax is not None:
        table_ax.remove()
        table_ax = None

    if "check_ax" in globals() and check_ax is not None:
        check_ax.remove()
        check_ax = None

    if "remove_button_ax" in globals() and remove_button_ax is not None:
        remove_button_ax.remove()
        remove_button_ax = None

    # Update table & UI safely
    update_table()
    fig.canvas.draw()

# Function to update the table and checkboxes
def update_table():
    global table_ax, check_ax, checkbuttons, remove_button_ax

    if "table_ax" in globals() and table_ax:
        table_ax.remove()
    if "check_ax" in globals() and check_ax:
        check_ax.remove()
    if "remove_button_ax" in globals() and remove_button_ax:
        remove_button_ax.remove()

    if not layer_labels:
        return

    # Create table for layers
    table_ax = fig.add_axes([0.85, 0.4, 0.12, 0.4])
    table_ax.axis("off")

    rows = [[label, "✔" if vis else "✘"] for label, vis in zip(layer_labels, visibility_flags)]
    
    if rows:
        table_ax.table(cellText=rows, colLabels=["Layer", "✓"], loc="center", cellLoc="center", bbox=[0, 0, 1, 1])

    # Create checkboxes
    check_ax = fig.add_axes([0.85, 0.1, 0.12, 0.2])
    check_ax.axis("off")
    
    checkbuttons = CheckButtons(check_ax, layer_labels, visibility_flags)
    checkbuttons.on_clicked(toggle_visibility)

    # Add "Remove Selected" button in the table area
    remove_button_ax = fig.add_axes([0.85, 0.03, 0.12, 0.05])
    remove_button = Button(remove_button_ax, 'Remove Selected')
    remove_button.on_clicked(remove_selected)

# Buttons for connecting points and clearing all
ax_button1 = plt.axes([0.7, 0.05, 0.1, 0.05])
btn_connect = Button(ax_button1, 'Connect')
btn_connect.on_clicked(connect_points)

ax_button2 = plt.axes([0.7, 0.12, 0.1, 0.05])
btn_clear = Button(ax_button2, 'Clear All')
btn_clear.on_clicked(clear_all)

update_table()  # Initialize the table
plt.show()