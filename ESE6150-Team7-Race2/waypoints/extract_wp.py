import pandas as pd
import numpy as np


df = pd.read_csv('waypoints/race2_3.csv')


n_points = 30

# use linspace to generate a set of indices that are uniformly spaced
indices = np.linspace(0, len(df) - 1, n_points, dtype=int)

# use these indices to extract the corresponding rows from the dataframe
uniform_points = df.iloc[indices]


uniform_points.to_csv('waypoints/key_wp_0401.csv', index=False)
