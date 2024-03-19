import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
import numpy as np

def read_circle_centers(file_path):
    """Read circle centers from a CSV file into a pandas DataFrame."""
    df = pd.read_csv(file_path)
    df = df.astype({'x': float, 'y': float, 'z': float, 'radius': float})
    return df[['x', 'y']].values

def plot_delaunay_triangulation(points, title='Delaunay Triangulation'):
    """Plot the Delaunay triangulation of a set of points."""
    tri = Delaunay(points)
    plt.figure(figsize=(10, 10))
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.title(title)
    plt.show()

def plot_convex_hull(points, title='Convex Hull'):
    """Plot the convex hull of a set of points."""
    hull = ConvexHull(points)
    plt.figure(figsize=(10, 10))
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.title(title)
    plt.show()

def inflate_convexhull(hull, points, inflation_distance=-1.5):
    """Inflate the convex hull by moving its vertices along the normals by a fixed distance."""
    inflated_vertices = np.zeros_like(points[hull.vertices])
    normals = np.zeros_like(inflated_vertices)

    # Calculate normals for each edge and normalize
    for i in range(len(hull.vertices)):
        next_index = (i + 1) % len(hull.vertices)
        edge_vector = points[hull.vertices[next_index]] - points[hull.vertices[i]]
        normal = np.array([-edge_vector[1], edge_vector[0]])  # Perpendicular to the edge
        normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
        normals[i] = normal

    # Calculate the normal for each vertex by averaging the normals of the adjacent edges
    for i, vertex in enumerate(hull.vertices):
        prev_normal = normals[i-1]
        current_normal = normals[i]
        vertex_normal = (prev_normal + current_normal) / 2
        vertex_normal = vertex_normal / np.linalg.norm(vertex_normal)  # Normalize the average normal

        # Move the vertex along the normal by a fixed distance
        inflated_vertices[i] = points[vertex] + vertex_normal * inflation_distance

    return inflated_vertices

def main():
    file_path = r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\circle_centers.csv"
    points = read_circle_centers(file_path)

    # Delaunay Triangulation for all points
    plot_delaunay_triangulation(points)

    # Divide the data into left and right sets
    left = points[points[:, 0] < 20]
    right = points[points[:, 0] > 20]

    # Delaunay Triangulation for left and right sets
    plot_delaunay_triangulation(left, 'Delaunay Triangulation - Left Side')
    plot_delaunay_triangulation(right, 'Delaunay Triangulation - Right Side')

    # Convex Hull and Inflated Convex Hull for both sets
    for side, side_points in [('Left Side', left), ('Right Side', right)]:
        hull = ConvexHull(side_points)
        inflated_points = inflate_convexhull(hull, side_points)
        
        # plot convex hull
        plot_convex_hull(side_points, f'Convex Hull - {side}')
        
        # plot inflated convex hull
        plt.figure(figsize=(10, 10))
        plt.fill(inflated_points[:, 0], inflated_points[:, 1], 'k-', alpha=0.2, edgecolor='black')
        plt.plot(side_points[:, 0], side_points[:, 1], 'o')
        plt.title(f'Inflated Convex Hull - {side}')
        plt.show()

if __name__ == '__main__':
    main()
