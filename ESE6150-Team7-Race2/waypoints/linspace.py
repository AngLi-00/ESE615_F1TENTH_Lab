import numpy as np
import pandas as pd
from os.path import expanduser
from time import gmtime, strftime

start_point = [2.71551735663049,-0.450954377019401]
end_point = [9.50172085636364,0.190707291140981]

n_points = 23



x = np.linspace(start_point[0], end_point[0], n_points)
y = np.linspace(start_point[1], end_point[1], n_points)

file = open(strftime('test_wp-%Y-%m-%d-%H-%M-%S',gmtime())+'.csv', 'w+')

for x, y in zip(x, y):
    print(x, y)
    # save the waypoints to a csv file
    file.write('%f, %f\n' % (x,y))

