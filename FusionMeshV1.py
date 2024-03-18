import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import Circle
import adsk.core, adsk.fusion, traceback, math

# generate a data structure of 10 x 10 x 10 of random numbers
rand_data = np.random.rand(10, 10, 10)  # 10 x 10 x 10 array of random numbers

# Define the spacing between points
spacing = 8.382

# create a coordinate grid for the data
x = np.arange(0, 10 * spacing, spacing)
y = np.arange(0, 10 * spacing, spacing)
z = np.arange(0, 10 * spacing, spacing)
X, Y, Z = np.meshgrid(x, y, z)










def main():


main()

