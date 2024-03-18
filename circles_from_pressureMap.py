# circles_from_pressureMap.py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist

def load_and_process_data(file_path):
    # Load the pressure data from CSV file
    foot_pressure = pd.read_csv(file_path).values

    # pad and rotate the pressure data
    foot_pressure = np.pad(foot_pressure, ((2, 2), (2, 2)), mode='constant', constant_values=0)  
    foot_pressure = np.rot90(foot_pressure, 2)
    return foot_pressure

def calculate_circle_centers(foot_pressure, spacing=8.382, min_circle_size=3, max_circle_size=15):
    # Process the foot pressure data similarly to before
    fp_rows, fp_cols = foot_pressure.shape
    x = np.arange(0, fp_cols * spacing, spacing)
    y = np.arange(0, fp_rows * spacing, spacing)
    interpolated_fp = RegularGridInterpolator((y, x), foot_pressure, method='linear')
    x_fine = np.linspace(0, (fp_cols - 1) * spacing, fp_cols * 10) 
    y_fine = np.linspace(0, (fp_rows - 1) * spacing, fp_rows * 10)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    points_to_interpolate = np.vstack((Y_fine.ravel(), X_fine.ravel())).T
    fine_fp = interpolated_fp(points_to_interpolate)
    normalized_fp_2d = (fine_fp.reshape(Y_fine.shape) - np.min(fine_fp)) / (np.max(fine_fp) - np.min(fine_fp))

    circles = []
    sorted_indices = np.dstack(np.unravel_index(np.argsort(-normalized_fp_2d.ravel()), normalized_fp_2d.shape))[0]
    for index in sorted_indices:
        i, j = index
        pressure = normalized_fp_2d[i, j]
        if pressure > 0:  # Ignore zero pressure values
            radius = min_circle_size + (pressure * (max_circle_size - min_circle_size))
            new_circle = {'x': x_fine[j], 'y': y_fine[i], 'radius': radius, 'z': 6}
            if not check_overlap(new_circle, circles):
                circles.append(new_circle)
    for circle in circles:
        circle['z'] = circle['z'] - circle['radius']
    return circles

def check_overlap(new_circle, circles):
    for circle in circles:
        distance = np.hypot(circle['x'] - new_circle['x'], circle['y'] - new_circle['y'])
        if distance < (circle['radius'] + new_circle['radius']):
            return True
    return False

if __name__ == '__main__':
    file_path = 'TK_footpressure.csv'
    foot_pressure = load_and_process_data(file_path)
    circles = calculate_circle_centers(foot_pressure)
    print("Circle centers:", circles)
    # out circles to csv
    df = pd.DataFrame(circles)
    # round all to 2 decimal places
    df = df.round(2)
    # divide radius by 10 to convert from mm to cm
    df['radius'] = df['radius'] / 10
    df.to_csv('circle_centers.csv', index=False)
    print("Circle centers saved to circle_centers.csv")
