# cutting_lattice.py
'''
This script reads the circle centers from a CSV file,

'''
import csv
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from skimage import measure
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Path to the CSV file
file_path = r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\circle_centers.csv"

# Read circle centers from the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Ensure correct data types
df = df.astype({'x': float, 'y': float, 'z': float, 'radius': float})

# Define grid for interpolation
grid_x, grid_y = np.mgrid[df['x'].min():df['x'].max():100j, df['y'].min():df['y'].max():100j]

# Interpolate z values on this grid
grid_z = griddata((df['x'], df['y']), df['z'], (grid_x, grid_y), method='cubic')

# Define the vertices of the mesh
vertices = np.array([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T

# Define faces based on the grid structure
faces = []
for j in range(grid_z.shape[1] - 1):
    for i in range(grid_z.shape[0] - 1):
        # Each square cell is divided into two triangles
        faces.append([i + j * grid_z.shape[0], i + 1 + j * grid_z.shape[0], i + 1 + (j + 1) * grid_z.shape[0]])  # Triangle 1
        faces.append([i + j * grid_z.shape[0], i + 1 + (j + 1) * grid_z.shape[0], i + (j + 1) * grid_z.shape[0]])  # Triangle 2

# Convert faces to vertices for plotting
faces_vertices = np.array([vertices[face] for face in faces])

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Adding the faces to the plot
mesh = Poly3DCollection(faces_vertices, alpha=0.5, edgecolor='k')
ax.add_collection3d(mesh)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Setting the limits for the axes
ax.set_xlim([df['x'].min(), df['x'].max()])
ax.set_ylim([df['y'].min(), df['y'].max()])
ax.set_zlim([df['z'].min(), df['z'].max()])

plt.show()

# Path to save the STL file
stl_file_path = "your_mesh.stl"

# Generate vertices for the mesh
vertices = np.array([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T

# Helper function to create faces
def create_faces(xi, yi, num_x):
    """Create two faces for the square defined by the (xi, yi) index."""
    return [
        [xi + yi * num_x, xi + 1 + yi * num_x, xi + 1 + (yi + 1) * num_x],
        [xi + yi * num_x, xi + 1 + (yi + 1) * num_x, xi + (yi + 1) * num_x]
    ]

# Number of points along the x and y axes
num_x, num_y = grid_z.shape

# Create faces
faces = []
for yi in range(num_y - 1):
    for xi in range(num_x - 1):
        faces.extend(create_faces(xi, yi, num_x))

# Create the mesh object with the correct size
your_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))

# Assign the vertices to the mesh object
for i, f in enumerate(faces):
    for j in range(3):  # Each face has 3 vertices
        your_mesh.vectors[i][j] = vertices[f[j], :]

# Save the mesh to an STL file
your_mesh.save(stl_file_path)
print(f"Mesh saved to {stl_file_path}")

