# cutting_lattice.py
'''
This script reads the circle centers from a CSV file,

'''
import pandas as pd
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np


# Path to the CSV file
file_path = r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\circle_centers.csv"



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