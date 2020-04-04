import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

image_qualities = np.loadtxt("cache/image_quailities.txt")
time_complexities = np.loadtxt("cache/time_complexities.txt")
# image_ratios = np.loadtxt("cache/image_ratios.txt")
# num_samples = 18
# v1 = np.linspace(1, 10, num_samples)
# v2 = np.linspace(1, 10, num_samples)

num_samples = 18
v1 = np.linspace(0.25, 4, num_samples)
v2 = np.linspace(0.25, 4, num_samples)
vv1, vv2 = np.meshgrid(v1, v2)

###########################################
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(vv1, vv2, image_qualities, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_title("image_qualities")
ax.set_xlabel("Horizontal")
ax.set_ylabel("Vertical")
###########################################
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(vv1, vv2, time_complexities, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_title("time_complexities")
ax.set_xlabel("Horizontal")
ax.set_ylabel("Vertical")
###########################################
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(vv1, vv2, image_qualities / time_complexities, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_title("$image\_qualities / exp(0.005 * time\_complexities)$")
ax.set_xlabel("Horizontal")
ax.set_ylabel("Vertical")
##########################################
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(vv1, vv2, image_qualities / time_complexities, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_title("$image\_qualities / time\_complexities$")
ax.set_xlabel("Horizontal")
ax.set_ylabel("Vertical")
##########################################
##########################################
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(vv1, vv2, image_qualities / np.exp(0.005 * vv1 * vv2), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_title("$image\_qualities / exp(0.005 * theta_{i,0} * theta_{i,1})$")
ax.set_xlabel("Horizontal")
ax.set_ylabel("Vertical")
##########################################
# ##########################################
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # Plot the surface.
# surf = ax.plot_surface(vv1, vv2, image_ratios, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# ax.set_title("Image Ratio")
# ax.set_xlabel("Horizontal")
# ax.set_ylabel("Vertical")
# ##########################################
plt.show()