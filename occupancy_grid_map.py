import numpy as np
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from tools import *
import data_extract

GRID_SCALE = 1
GRID_SIZE = (400, 450)

data = data_extract.read_data_file("Extracted States & Lidar Data.txt")
row = data.shape[0]

poses = np.zeros((row, 3))
scans = np.zeros((row, len(data[0]) - 3))

for i in range(row):
    poses[i] = data[i][0:3]
    scans[i] = (data[i][3:])

delta = np.array([250, 160, 50])

assert(len(poses) == len(scans))

grid = np.zeros(GRID_SIZE)
print(grid.shape)
bar = ProgressBar(len(poses))
for pose, scan in bar(zip(poses + delta, scans)):
    points = rotate(convert2xy(scan), pose[2]) + pose[:2]
    subgrid = convert2map(pose, points, GRID_SCALE, GRID_SIZE, 0.0005)
    grid += np.log(subgrid/(1-subgrid))

grid = 1/(1+np.exp(-grid))
print(grid)
extent = [
    -delta[0], GRID_SIZE[0]*GRID_SCALE-delta[0],
    -delta[1], GRID_SIZE[1]*GRID_SCALE-delta[1],
]
plt.imshow(grid.T[::-1], vmin=0, vmax=1, cmap=plt.cm.Greys, extent=extent)
plt.plot(poses[:, 0], poses[:, 1], label="Vehicle trajectory")
plt.xlabel("X, m")
plt.ylabel("Y, m")
plt.legend(loc=4)
plt.grid()
plt.tight_layout()
plt.savefig("OGM.png", dpi=300)
plt.show()
