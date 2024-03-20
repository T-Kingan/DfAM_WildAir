
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist

import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
import numpy as np
from stl import mesh


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

def check_overlap(new_circle, circles):   # check if the new circle overlaps with any existing circles - used in calculate_circle_centers
    for circle in circles:
        distance = np.hypot(circle['x'] - new_circle['x'], circle['y'] - new_circle['y'])
        if distance < (circle['radius'] + new_circle['radius']):
            return True
    return False


def inflate_convexhull(hull, points, inflation_distance=-1.5):
    """Inflate the convex hull by moving its vertices along the normals by a fixed distance."""
    # Points without z for computation, keep original for z values
    points_2d = points[:, :2]
    inflated_vertices_2d = np.zeros_like(points_2d[hull.vertices])
    normals = np.zeros_like(inflated_vertices_2d)

    # Calculate normals for each edge and normalize
    for i in range(len(hull.vertices)):
        next_index = (i + 1) % len(hull.vertices)
        edge_vector = points_2d[hull.vertices[next_index]] - points_2d[hull.vertices[i]]
        normal = np.array([-edge_vector[1], edge_vector[0]])  # Perpendicular to the edge
        normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
        normals[i] = normal

    # Calculate the normal for each vertex by averaging the normals of the adjacent edges
    for i, vertex in enumerate(hull.vertices):
        prev_normal = normals[i-1]
        current_normal = normals[i]
        vertex_normal = (prev_normal + current_normal) / 2
        vertex_normal = vertex_normal / np.linalg.norm(vertex_normal)  # Normalize the average normal

        # Move the vertex along the normal by a fixed distance in 2D
        inflated_vertices_2d[i] = points_2d[vertex] + vertex_normal * inflation_distance

    
    # set radius values of inflated points to 0
    inflated_vertices = np.column_stack((inflated_vertices_2d, np.full(len(inflated_vertices_2d), 0)))
    # Set z value of inflated points to 0.3
    inflated_vertices = np.column_stack((inflated_vertices, np.full(len(inflated_vertices), 0.3)))

    return inflated_vertices

def create_stl_from_delaunay(points, filename='output_mesh.stl'):
    # Assume points is in the format [x, y, radius, z]
    # Extract the x, y, and z values, ignoring the radius
    xyz_points = points[:, [0, 1, 3]]  # Select columns for x, y, and z, excluding radius

    # Perform Delaunay triangulation on the x and y values only
    tri = Delaunay(xyz_points[:, :2])  # Use just the x and y for triangulation

    # Create an empty mesh with the correct number of faces
    output_mesh = mesh.Mesh(np.zeros(tri.simplices.shape[0], dtype=mesh.Mesh.dtype))

    # Fill the mesh with the Delaunay triangles, using x, y, and z coordinates
    for i, f in enumerate(tri.simplices):
        for j in range(3):
            # Use f[j] as index to pull the correct [x, y, z] point for each vertex of the triangle
            output_mesh.vectors[i][j] = xyz_points[f[j], :]

    # Save the mesh to an STL file
    output_mesh.save(filename)

def check_overlap_numpy(new_circle, existing_circles):
    """
    Checks if a new circle overlaps with any of the existing circles.
    
    Parameters:
    - new_circle: A numpy array representing the new circle to check, with format [x, y, radius, z].
    - existing_circles: A numpy array of existing circles, each with format [x, y, radius, z].
    
    Returns:
    - True if there is an overlap with any existing circle, False otherwise.
    """
    # Extract x, y coordinates and radii from new_circle and existing_circles
    new_x, new_y, new_radius, _ = new_circle
    existing_xs = existing_circles[:, 0]
    existing_ys = existing_circles[:, 1]
    existing_radii = existing_circles[:, 2]
    
    # Calculate the distance between the new circle and all existing circles
    distances = np.sqrt((existing_xs - new_x)**2 + (existing_ys - new_y)**2)
    
    # Check for overlap: if the distance between centers is less than the sum of radii, circles overlap
    overlaps = distances < (new_radius + existing_radii)
    
    # If any overlap, return True
    return np.any(overlaps)

def add_additional_circles(inflated_points, side_points, additional_radius=0.3, grid_spacing=1):
    # Create a grid of points within the bounds of the inflated_points
    min_x, min_y = np.min(inflated_points[:, :2], axis=0)
    max_x, max_y = np.max(inflated_points[:, :2], axis=0)
    x_grid = np.arange(min_x, max_x, grid_spacing)
    y_grid = np.arange(min_y, max_y, grid_spacing)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T

    # Initialize a list to collect new circles
    new_circles = []

    # Combine existing and new circles to check for overlap
    combined_circles = np.array(side_points)  # Start with existing circles

    for point in grid_points:
        new_circle = np.array([point[0], point[1], additional_radius, 0])  # Structure as an array to match side_points
        
        # Now check_overlap_numpy checks against both existing and so-far added new circles
        if not check_overlap_numpy(new_circle, combined_circles) and is_inside_boundary(point, inflated_points):
            new_circles.append(new_circle)  # Collect new circles here
            combined_circles = np.vstack([combined_circles, new_circle])  # Update the combined list

    #return combined_circles
    return new_circles

def is_inside_boundary(point, boundary_points):
    """
    Checks if a point is inside the boundary defined by the given points. (using x & y only)
    """
    # only x and y coordinates are used
    point = point[:2]
    boundary_points = boundary_points[:, :2]
    hull = ConvexHull(boundary_points)
    new_hull = Delaunay(boundary_points[hull.vertices])
    return new_hull.find_simplex(point) >= 0

def interpolate_z_values(surface_points):
    """
    Creates an interpolation function for the z-values over the surface defined by the surface points.
    
    Parameters:
    - surface_points: A numpy array of points on the cutting surface, with format [x, y, z].
    
    Returns:
    - A function that takes x, y coordinates and returns the interpolated z-value.
    """
    # Separate x, y, and z coordinates
    points = surface_points[:, :2]  # x, y coordinates
    values = surface_points[:, 2]   # z coordinates
    
    # Create a griddata interpolator function
    def interpolator(x, y):
        return griddata(points, values, (x, y), method='linear')
    
    return interpolator

if __name__ == '__main__':
    file_path = 'TK_footpressure.csv'
    foot_pressure = load_and_process_data(file_path)
    circles = calculate_circle_centers(foot_pressure) # datatype: list of dictionaries
    print("Circle centers:", circles)

    # # plot the circles
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # for circle in circles:
    #     ax.add_patch(plt.Circle((circle['x'], circle['y']), circle['radius'], color='r', fill=False))
    # ax.set_xlim(0, 450)
    # ax.set_ylim(0, 450)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()


    # out circles to csv
    df = pd.DataFrame(circles) # datatype: pandas dataframe
    # divide all values by 10 to convert to cm
    df = df / 10
    # round all to 2 decimal places
    df = df.round(2)
    # print the df
    print(df)

    # save to csv
    df.to_csv("circle_centers.csv", index=False)
    print("Padding circle centers saved to circle_centers.csv")

    # convert df to numpy.ndarray
    circle_points = df.to_numpy() # datatype: numpy.ndarray - cm values
    print("Circle centers as numpy.ndarray:", circle_points)

    # Divide the circle points into left and right sets based on x value, maintain z value
    left = circle_points[circle_points[:, 0] < 20]
    #print("left", left)
    right = circle_points[circle_points[:, 0] > 20]

    # Convex Hull, Inflated points and cutting surface points for left and right sides
    for side, side_points in [('Left Side', left), ('Right Side', right)]:
        hull = ConvexHull(side_points[:, :2])  # Use only x, y for computation
        inflated_points = inflate_convexhull(hull, side_points) # create boundary points (radius = 0) # datatype: numpy.ndarray
        
        print("inflated_points", inflated_points)
        # side_points: datatype: numpy.ndarray

        # Add additional circles inside the boundary
        new_circles = add_additional_circles(inflated_points, side_points, additional_radius=0.2, grid_spacing=0.1)

        # plot additional circles
        fig, ax = plt.subplots()
        for circle in side_points:
            ax.add_patch(plt.Circle((circle[0], circle[1]), circle[2], color='r', fill=False))
        for circle in inflated_points:
            ax.add_patch(plt.Circle((circle[0], circle[1]), circle[2], color='b', fill=False))
        for circle in new_circles:
            ax.add_patch(plt.Circle((circle[0], circle[1]), circle[2], color='g', fill=False))
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


        # Add the inflated points to the original points, ensuring inflated points have z=0
        #inflated_points = np.concatenate((side_points, inflated_points), axis=0) # Change to 0.3
        cutting_surface_points = np.concatenate((side_points, inflated_points), axis=0)

        print("cutting_surface_points", cutting_surface_points)

        # Delaunay Triangulation for the inflated points in x and y
        tri = Delaunay(cutting_surface_points[:, :2])  # Ensure this is 2D # What is it?

        print("tri.simplices", tri.simplices)

        # Plot the Delaunay triangulation of the inflated points
        plt.figure(figsize=(10, 10))
        plt.triplot(cutting_surface_points[:, 0], cutting_surface_points[:, 1], tri.simplices)
        plt.plot(cutting_surface_points[:, 0], cutting_surface_points[:, 1], 'o')
        plt.title(f'Delaunay Triangulation - Inflated {side}')
        plt.show()

        # does delaunay in the function
        create_stl_from_delaunay(cutting_surface_points, f'delaunay_mesh_{side}.stl')
        print("STL file created")

        # Interpolate z-values over the cutting surface
        z_interpolator = interpolate_z_values(cutting_surface_points)

        #for circle in new_circles:
            # find the z value of the circle






# new circles?