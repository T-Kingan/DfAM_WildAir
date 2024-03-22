import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# Load the pressure data
file_path = 'selected_foot_pressure.csv'
foot_pressure = pd.read_csv(file_path).values
fp_rows, fp_cols = foot_pressure.shape  # Get the number of rows and columns
spacing = 8.382

# Create the x and y coordinates

