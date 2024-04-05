import matplotlib.pyplot as plt
# import cvxpy as cp
import numpy as np
import scipy as sp
import datetime
import os

current_path = os.path.dirname(os.path.realpath(__file__))

def calculate_normals_2d(points):
    """
    Calculate normal vectors to a list of 2D points.
    
    Parameters:
        points (ndarray): Array of shape (N, 2) containing 2D points.
    
    Returns:
        ndarray: Array of shape (N, 2) containing normal vectors.
    """
    n_points = np.vstack([points, points[0].reshape(1, 2)])
    # Calculate differences between points
    differences = np.diff(n_points, axis=0)
    
    # Swap x and y coordinates and change the sign of one coordinate to get normals
    normals = np.array([-differences[:, 1], differences[:, 0]]).T
    
    # Normalize normal vectors
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    
    return normals

def get_xy_from_file(filename):
    data = np.genfromtxt(current_path + '/' + filename, delimiter=',', skip_header=1,
                        dtype=[('x', float), ('y', float)])

    # Access the loaded data
    data_x =  data['x']
    data_y =  data['y']
    # points = np.vstack((data_x, data_y)).T
    return data_x, data_y

def save_xy_to_csv_numpy(filename, x, y):
    """
    Save xy points to a CSV file using numpy.
    
    Parameters:
        filename (str): Name of the CSV file to save.
        x (list or array): List or array of x-coordinates.
        y (list or array): List or array of y-coordinates.
    """
    xy_points = np.vstack((x, y)).T
    np.savetxt(current_path + '/' + filename, xy_points, delimiter=',', comments='')


if __name__ == '__main__':

    file_path = 'key_wp_0401_5_realrace.csv'
    track_path = 'race2.png'

    save_data = True

    # Access the CSV Data
    data_x, data_y = get_xy_from_file(file_path)

    # Get Normals
    points = np.vstack((data_x, data_y)).T
    normals = calculate_normals_2d(points)
    
    # Select and Plot Points & Normals
    x = data_x #[::2]
    y = data_y #[::2]
    normals = normals #[::2]
    # plt.figure()
    # plt.quiver(x, y, normals[:, 0], normals[:, 1], angles='xy')


    # Append the starting x,y coordinates (modified to remove points)
    x = np.r_[x[:], x[0]]
    y = np.r_[y[:], y[0]]


    # Fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    tck, u = sp.interpolate.splprep([x, y], s=0, per=True)

    # Evaluate the spline fits for 100 evenly spaced distance values
    xi, yi = sp.interpolate.splev(np.linspace(0, 1, 500), tck)

    # Plot the result
    fig, ax = plt.subplots(1, 1)

    # Plot the race map
    image = plt.imread(current_path + '/' + track_path) 
    extent = [-7.06, -7.06 + image.shape[1] * .05, -4.8, -4.8 + image.shape[0] * .05]
    ax.imshow(image, extent=extent, cmap='gray')

    # Plot spline path
    ax.plot(xi, yi, '-b')

    # # Plot original data
    # data_x = np.r_[data_x, data_x[0]]
    # data_y = np.r_[data_y, data_y[0]]
    # ax.plot(data_x, data_y, 'og')

    # Plot points used to make spline
    ax.plot(x, y, 'or')
    ax.plot(x[0], y[0], 'og')
    
    if save_data:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        save_xy_to_csv_numpy('Modified_Points_' + timestamp + ".csv", xi, yi)

    plt.show()

#################### Scrap Optimization Code #############################
# t_lwidth = 2
# t_rwidth = 2

# alpha = cp.Variable(normals.shape[0])
# x_p = cp.Variable((normals.shape[0], 2))
# dx_p = cp.Variable((normals.shape[0], 2))

# constraints = []

# # Alpha Constraints
# constraints += [
#     x_p == points + normals * alpha,
#     alpha < np.ones(normals.shape) * t_lwidth,
#     alpha > np.ones(normals.shape) * t_rwidth
# ]

# # Derivative Constraints
# constraints += [dx_p == cp.diff(x_p, axis=1)]








