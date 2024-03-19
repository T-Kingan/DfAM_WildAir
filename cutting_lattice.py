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
#from vtkplotlib import plot as vtk_plot
from vtkplotlib import mesh_plot, show

# Path to the CSV file
file_path = r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\circle_centers.csv"

# Read circle centers from the CSV file into a pandas DataFrame
circle_centers = []
with open(file_path, mode='r', newline='') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        circle_centers.append(row)

df = pd.DataFrame(circle_centers)
df = df.astype({'x': float, 'y': float, 'z': float, 'radius': float})

# Define grid for interpolation
grid_x, grid_y = np.mgrid[df['x'].min():df['x'].max():100j, df['y'].min():df['y'].max():100j]

# Interpolate z values on this grid
grid_z = griddata(df[['x', 'y']], df['z'], (grid_x, grid_y), method='cubic')

# Create a mesh from the interpolated surface, e.g., by finding contours
contours = measure.find_contours(grid_z, 0.5)  # Adjust level based on your data

# Convert contours to mesh (simplified example, more steps may be needed based on your requirements)
vertices = np.empty((0, 3))
faces = []
index = 0
for contour in contours:
    contour_length = len(contour)
    vertices = np.vstack([vertices, np.hstack([contour, np.zeros((contour_length, 1))])])
    faces.extend([[i, i + 1, i + 2] for i in range(index, index + contour_length - 2)])
    index += contour_length

# Create a mesh object and save to an STL file
your_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        your_mesh.vectors[i][j] = vertices[f[j], :]

# Save mesh to file
#your_mesh.save('your_mesh.stl')

mesh_plot(your_mesh)
show()
