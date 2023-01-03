import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

''' 3D visualization of the clusters positions for the simulator'''
# grapes_cords = [[1.1, 2.9, 1.35],  # 1
#                 [1.2, 2.08, 1.5],  # 2 above the split
#                 [0.7, 1.42, 1.8],  # 3 above the split
#                 [1.2, 1, 1.2],  # 4
#                 [0.95, -0.15, 1.28],  # 5
#                 [1.15, -1.53, 1.47],  # 6
#                 [1.05, -1.42, 1.6],  # 7 above the split
#                 [1.2, -2.15, 1.8],  # 8 above the split
#                 [0.85, -3.2, 1.7],  # 9 above the split
#                 [1, -4, 1.5]]  # 10

grapes_cords = [[1.2, 0, 1.2],
                [1.05, 0, 1.28],
                [1.1, 0, 1.35],
                [0.9, 0, 1.47],
                [1.2, 0, 1.5],
                [1.05, 0, 1.6],
                [0.85, 0, 1.7],
                [1.1, 0, 1.75],
                [1.2, 0.15, 1.8],
                [0.7, 0, 1.8]]

npcords = np.array(grapes_cords)
xs = npcords[:, 0]
ys = npcords[:, 1]
zs = npcords[:, 2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim3d(0, 1.2)
ax.set_ylim3d(-5.5, 5.5)
ax.set_zlim3d(1.1, 1.8)

ax.set_title("Clusters positions in the vineyard")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

ax.scatter(xs, ys, zs, color='purple', label="Clusters")

Xc = [0, 0, 0, 0, 0.1]
Yc = [0, 0, 0.7, 0.7, 0.7]
Zc = [1, 1.1, 1.1, 1.4, 1.4]
ax.plot3D(Xc, Yc, Zc, '-o', linewidth=7, markersize=10, label="Robot")

X_wire = [1.2,1.2, 0.8]
Y_wire = [0, 0, 0]
Z_wire = [1.1,1.5, 1.8]
ax.plot3D(X_wire, Y_wire, Z_wire, color='black', linewidth=7, markersize=7, label="wire")

plt.legend()
plt.show()
