'''
Takes in the foot pressure data as a preformated csv
- Plots various pressure maps
- Final output is a map of circles

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import Circle



# Load the pressure data from CSV file
file_path = 'TK_footpressure.csv'
foot_pressure = pd.read_csv(file_path).values

# pad the pressure data with two rows and two columns of zeros on all sides
foot_pressure = np.pad(foot_pressure, ((2, 2), (2, 2)), mode='constant', constant_values=0)  
# rotate the pressure data 180 degrees
foot_pressure = np.rot90(foot_pressure, 2)

print("foot_pressure: ", foot_pressure)

# Get the shape of the foot pressure data
fp_rows, fp_cols = foot_pressure.shape

# Define the spacing between sensors
spacing = 8.382

# Create a grid of coordinates for the original pressure data
x = np.arange(0, fp_cols * spacing, spacing)
y = np.arange(0, fp_rows * spacing, spacing)

# Create an interpolation function based on the original pressure data grid
interpolated_fp = RegularGridInterpolator((y, x), foot_pressure, method='linear') # method that can interpolate the pressure data

# Define a finer grid for interpolation
# 10 units = 1 mm
x_fine = np.linspace(0, (fp_cols - 1) * spacing, fp_cols * 10) 
print("x_fine:", x_fine)
y_fine = np.linspace(0, (fp_rows - 1) * spacing, fp_rows * 10)
print("y_fine:", y_fine)

# Create meshgrid for the fine grid
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

# Flatten the fine meshgrid and prepare for interpolation
points_to_interpolate = np.vstack((Y_fine.ravel(), X_fine.ravel())).T

# Interpolate pressure data
fine_fp = interpolated_fp(points_to_interpolate)

# Reshape the interpolated data to 2D array for plotting
fine_fp_2d = fine_fp.reshape(Y_fine.shape)  # 2D array of pressure values that correspond to the fine grid

# Plotting
plt.figure(figsize=(10, 15))
plt.contourf(X_fine, Y_fine, fine_fp_2d, levels=100, cmap='viridis')
plt.colorbar(label='Pressure (arbitrary units)')
plt.title('Interpolated Pressure Data')
plt.xlabel('X coordinate (mm)')
plt.ylabel('Y coordinate (mm)')
plt.show()

# # plot a histogram of the interpolated pressure data removing 0 values
# plt.hist(fine_fp[fine_fp > 0], bins=50, color='blue', alpha=0.7)
# plt.title('Interpolated Pressure Data Histogram')
# plt.xlabel('Pressure (arbitrary units)')
# plt.ylabel('Frequency')

# plt.show()

# flatten the pressure data
flat_fp = fine_fp.flatten()

# number of segments
n = 10
# calculate the quantile thresholds (excluding 0 values)
quantile_thresholds = np.percentile(flat_fp[flat_fp > 0], np.linspace(0, 100, n + 1))

# print the quantile thresholds rounded to 2 decimal places
print(np.round(quantile_thresholds, 2))


# plot the pressure data with the segmented thresholds
plt.figure(figsize=(10, 15)) # 
plt.contourf(X_fine, Y_fine, fine_fp_2d, levels=quantile_thresholds, cmap='viridis')
plt.colorbar(label='Pressure (arbitrary units)')
plt.title('Segmented Pressure Data')
plt.xlabel('X coordinate (mm)')
plt.ylabel('Y coordinate (mm)')
plt.show()

# --- Testing ---

# `normalized_fp_2d` is the 2D array of pressure values normalized between 0 and 1
normalized_fp_2d = (fine_fp_2d - np.min(fine_fp_2d)) / (np.max(fine_fp_2d) - np.min(fine_fp_2d))

# Define the pressure threshold for binary image
pressure_threshold = 0.5  # Replace with your chosen threshold

# Create binary image based on threshold
binary_image = np.where(fine_fp_2d >= pressure_threshold, 1, 0)

# divide the binary image in 2 along the x axis
binary_image_left = binary_image[:, :binary_image.shape[1]//2]
binary_image_right = binary_image[:, binary_image.shape[1]//2:]

# perform a closing operation on the binary images
# closing is a dilation followed by an erosion
# this is to remove any small holes in the binary image
# that are not part of the foot
from scipy.ndimage import binary_closing
binary_image_left = binary_closing(binary_image_left)
binary_image_right = binary_closing(binary_image_right)

# join the binary images back together
binary_image = np.concatenate((binary_image_left, binary_image_right), axis=1)

# Plot the binary image
plt.figure(figsize=(10, 15))
plt.imshow(binary_image, cmap='gray', origin='lower')
plt.colorbar(label='Pressure (binary)')
plt.title('Binary Pressure Map')
plt.xlabel('X coordinate (mm)')
plt.ylabel('Y coordinate (mm)')
plt.show()

# # Assume `x_fine` and `y_fine` are the fine grid coordinates as before

# # Function to check if a new circle overlaps with any existing circle
# def check_overlap(new_circle, circles):
#     for circle in circles:
#         distance = np.hypot(circle['x'] - new_circle['x'], circle['y'] - new_circle['y'])
#         if distance < (circle['radius'] + new_circle['radius']):
#             return True
#     return False

# # Initialize list to keep track of circles
# circles = []

# # Define minimum and maximum circle sizes
# min_circle_size = 3  # replace with your chosen minimum
# max_circle_size = 15 #spacing / 2  # the chosen maximum, as before
# extra_circle_size = 1.5 

# # Iterate over the pressure values, starting with the highest pressure
# sorted_indices = np.dstack(np.unravel_index(np.argsort(-normalized_fp_2d.ravel()), normalized_fp_2d.shape))[0]

# # Iterate over the pressure values, starting with the highest pressure
# for index in sorted_indices:
#     i, j = index
#     pressure = normalized_fp_2d[i, j]
#     if pressure > 0:  # Ignore zero pressure values
#         # Map pressure to the new circle size range
#         radius = min_circle_size + (pressure * (max_circle_size - min_circle_size))
#         new_circle = {'x': x_fine[j], 'y': y_fine[i], 'radius': radius}
        
#         # Check if the new circle overlaps with any existing circle
#         if not check_overlap(new_circle, circles):
#             circles.append(new_circle)

# print(circles)

# # export the circles to a CSV file
# circles_df = pd.DataFrame(circles)
# circles_df.to_csv('circles.csv', index=False)

# # Plot the circles
# fig, ax = plt.subplots(figsize=(10, 15))
# for circle in circles:
#     ax.add_patch(Circle((circle['x'], circle['y']), circle['radius'], fill=False))

# # Adjust the plot limits and aspect ratio
# ax.set_xlim([0, np.max(x_fine)])
# ax.set_ylim([0, np.max(y_fine)])
# ax.set_aspect('equal')
# plt.title('Packed Circles According to Pressure Data')
# plt.xlabel('X coordinate (mm)')
# plt.ylabel('Y coordinate (mm)')
# plt.show()