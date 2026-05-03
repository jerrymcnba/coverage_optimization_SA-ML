import numpy as np
import random
import matplotlib.pyplot as plt
import lightgbm as lgb


w=10
h=5
num_circles=24
circles = [(x := random.uniform(0,w), y := random.uniform(0,h), r := 1) for _ in range (num_circles)]

uncovered_points=0
num_points=150 

def plot_circles(circles):
  fig, ax = plt.subplots()

  rectangle = plt.Rectangle((0, 0), w, h, linewidth=2, edgecolor='black', facecolor='none')
  ax.add_patch(rectangle)

  for x,y,r in circles:
    circle=plt.Circle((x,y),r,color='blue',fill=False,alpha=0.5)
    ax.add_patch(circle)

  ax.set_xlim(0,w)
  ax.set_ylim(0,h)
  ax.set_aspect('equal')

  plt.show()


grid_x, grid_y = np.meshgrid(
    np.linspace(0, w, num_points), 
    np.linspace(0, h, num_points) 
    ) 

grid_points = np.stack( 
    [grid_x.ravel(), grid_y.ravel()], axis=1 
    ) 