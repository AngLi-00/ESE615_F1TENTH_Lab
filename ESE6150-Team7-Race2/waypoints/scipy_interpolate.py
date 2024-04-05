import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

df = pd.read_csv('waypoints/key_wp_0401.csv',header=None, names=['x', 'y'])

x = df['x'].values
y = df['y'].values

# Generate spline trajectory
tck, u = splprep([x, y], s=0, per=True)
u_new = np.linspace(0, 1, num=500)
spline_coords = splev(u_new, tck)


# Plot the spline trajectory
plt.plot(spline_coords[0], spline_coords[1])
plt.scatter(x, y, color='red', label='Waypoints')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Spline Trajectory')
plt.legend()
plt.show()

# Save spline coordinates to a CSV file
spline_coords_df = pd.DataFrame({'x': spline_coords[0], 'y': spline_coords[1]})
spline_coords_df.to_csv('waypoints/spline_coords.csv', index=False, header=False)



