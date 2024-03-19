# cutting_lattice.py
'''
This script reads the circle centers from a CSV file,

'''
import pandas as pd
from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt
import numpy as np


# Path to the CSV file
file_path = r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\circle_centers.csv"

# Function to inflate convex hull vertices
def inflate_convexhull(hull, points, inf=1.5):
    centroid = np.mean(points[hull.vertices], axis=0)
    inflated_vertices = []
    for vertex in points[hull.vertices]:
        direction = vertex - centroid
        inflated_vertex = centroid + direction * inf
        inflated_vertices.append(inflated_vertex)
    return np.array(inflated_vertices)

# Read circle centers from the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Ensure correct data types
df = df.astype({'x': float, 'y': float, 'z': float, 'radius': float})

# Extract XY coordinates
points = df[['x', 'y']].values

# Perform Delaunay triangulation
tri = Delaunay(points)  # Create the Delaunay triangulation


# Plotting the triangulation
plt.figure(figsize=(10, 10)) 
plt.triplot(points[:, 0], points[:, 1], tri.simplices) 
plt.plot(points[:, 0], points[:, 1], 'o')
plt.show()

# divide the data into 2 down the centre of x axis
left = points[points[:, 0] < 20]   # 
right = points[points[:, 0] > 20]

# Perform Delaunay triangulation
tri_left = Delaunay(left)  # Create the Delaunay triangulation
tri_right = Delaunay(right)  # Create the Delaunay triangulation

# Plotting the triangulation
plt.figure(figsize=(10, 10))
plt.triplot(left[:, 0], left[:, 1], tri_left.simplices)
plt.plot(left[:, 0], left[:, 1], 'o')
plt.show()

plt.figure(figsize=(10, 10))
plt.triplot(right[:, 0], right[:, 1], tri_right.simplices)
plt.plot(right[:, 0], right[:, 1], 'o')
plt.show()

# Extract XY coordinates
points = df[['x', 'y']].values

# Divide the data into 2 down the centre of the x-axis
left = points[points[:, 0] < 20]
right = points[points[:, 0] > 20]

# Find the convex hull for the left set
hull_left = ConvexHull(left)
inflated_left = inflate_convexhull(hull_left, left)

# Find the convex hull for the right set
hull_right = ConvexHull(right)
inflated_right = inflate_convexhull(hull_right, right)

# Plotting the convex hull for the left set
plt.figure(figsize=(10, 10))
for simplex in hull_left.simplices:
    plt.plot(left[simplex, 0], left[simplex, 1], 'k-')  # 'k-' is for black line
plt.plot(left[:, 0], left[:, 1], 'o')
plt.title('Convex Hull - Left Side')
plt.show()

# Plotting the convex hull for the right set
plt.figure(figsize=(10, 10))
for simplex in hull_right.simplices:
    plt.plot(right[simplex, 0], right[simplex, 1], 'k-')  # 'k-' is for black line
plt.plot(right[:, 0], right[:, 1], 'o')
plt.title('Convex Hull - Right Side')
plt.show()

# Plotting the inflated convex hull for the left set
plt.figure(figsize=(10, 10))
for i, simplex in enumerate(hull_left.simplices):
    plt.plot(inflated_left[[i, (i+1) % len(inflated_left)], 0], inflated_left[[i, (i+1) % len(inflated_left)], 1], 'k-')
plt.plot(left[:, 0], left[:, 1], 'o')
plt.title('Inflated Convex Hull - Left Side')
plt.show()

# Plotting the inflated convex hull for the right set
plt.figure(figsize=(10, 10))
for i, simplex in enumerate(hull_right.simplices):
    plt.plot(inflated_right[[i, (i+1) % len(inflated_right)], 0], inflated_right[[i, (i+1) % len(inflated_right)], 1], 'k-')
plt.plot(right[:, 0], right[:, 1], 'o')
plt.title('Inflated Convex Hull - Right Side')
plt.show()
