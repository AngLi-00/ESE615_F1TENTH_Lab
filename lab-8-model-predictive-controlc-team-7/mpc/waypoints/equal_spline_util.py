import numpy as np
from scipy.interpolate import splprep, splev
from scipy.integrate import quad

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def sample_spline_with_distance(x, y, desired_distance):
    tck, u = splprep([x, y], s=0, per=True)

    # Compute arc length
    arc_length, _ = quad(lambda t: np.sqrt(splev(t, tck, der=1)[0]**2 + splev(t, tck, der=1)[1]**2), u[0], u[-1])

    # Determine number of points
    num_points = int(arc_length / desired_distance)

    # Sample points
    points = []
    vels = []
    prev_point = None
    for i in range(num_points):
        t = i * (u[-1] - u[0]) / num_points
        point = np.zeros(3)
        point[:2] = np.array(splev(t, tck))

        vel = np.array(splev(t, tck, 1))
        point[2] = np.arctan2(vel[1], vel[0])
        # if prev_point is None or euclidean_distance(point, prev_point) >= desired_distance:
        points.append(point)
        vels.append(vel)
        # prev_point = point

    return np.array(points), np.array(vels)

def average_distance_between_points(points):
    # Calculate the difference between consecutive points
    differences = np.diff(points[:, :2], axis=0)
    
    # Calculate the Euclidean distance for each difference vector
    distances = np.linalg.norm(differences, axis=1)
    
    # Calculate the average distance
    average_distance = np.mean(distances)
    max_dist = np.max(distances)
    min_dist = np.min(distances)
    
    return average_distance, min_dist, max_dist, distances

# Example usage
if __name__ == "__main__":
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 0, 1, 0])

    desired_distance = 1.0

    sampled_points = sample_spline_with_distance(x, y, desired_distance)
    print(sampled_points)
