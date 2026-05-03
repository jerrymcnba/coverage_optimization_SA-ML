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

def uncovered_points_count(circles):
    centers = np.array([(cx, cy) for cx, cy, _ in circles])   
    radii = np.array([r for _, _, r in circles])              

    
    diff = grid_points[:, None, :] - centers[None, :, :]     
    dist2 = np.sum(diff ** 2, axis=2)                         

    covered = np.any(dist2 <= radii**2, axis=1)              
    return np.sum(~covered)


def covered_percen(circles):
  covered_percen = 1 - (uncovered_points_count(circles) / (num_points**2))
  return covered_percen

def overlap_penalty(circles):
    penalty = 0
    centers = np.array([(cx, cy) for cx, cy, _ in circles])
    radii = np.array([r for _, _, r in circles])

    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            dist = np.linalg.norm(centers[i] - centers[j])
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                penalty += overlap**2   

    return penalty

def total_cost(circles, alpha=1.0, beta=5.0):
    uncovered = uncovered_points_count(circles)
    overlap = overlap_penalty(circles)
    return alpha * uncovered + beta * overlap

def local_density(circles, idx):
    cx, cy, r = circles[idx]
    count = 0
    for i, (x, y, _) in enumerate(circles):
        if i == idx:
            continue
        if np.linalg.norm([cx - x, cy - y]) < 2*r:
            count += 1
    return count

def extract_features(circles, new_circles, modified_idx, dx, dy):
    cx, cy, r = circles[modified_idx]
    new_cx, new_cy, _ = new_circles[modified_idx]

    
    dist_left = cx
    dist_right = w - cx
    dist_bottom = cy
    dist_top = h - cy

    density = local_density(circles, modified_idx)

    return [
        cx, cy,
        dx, dy,
        dist_left, dist_right, dist_bottom, dist_top,
        density
        ]


def collect_training_data(circles, samples=5000):
    X = []
    y = []

    current = circles.copy()

    for _ in range(samples):
        new = current.copy()
        idx = random.randint(0, num_circles - 1)
        cx, cy, r = new[idx]

        dx = random.uniform(-0.5, 0.5)
        dy = random.uniform(-0.5, 0.5)

        new_cx = np.clip(cx + dx, 0, w)
        new_cy = np.clip(cy + dy, 0, h)
        new[idx] = (new_cx, new_cy, r)

        old_cost = total_cost(current)
        new_cost = total_cost(new)

        X.append(extract_features(current, new, idx, dx, dy))
        y.append(new_cost - old_cost)

        current = new

    return np.array(X), np.array(y)


def train_lgb_model(X, y):
    train_data = lgb.Dataset(X, label=y)

    params = {
        'objective': 'regression',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbosity': -1
        }

    model = lgb.train(params, train_data, num_boost_round=50)
    return model


def modify_circles_with_info(circles):
    max_step = 0.5
    new_circles = circles.copy()

    modified = random.randint(0, num_circles - 1)
    cx, cy, r = new_circles[modified]

    dx = random.uniform(-max_step, max_step)
    dy = random.uniform(-max_step, max_step)

    new_cx = np.clip(cx + dx, 0, w)
    new_cy = np.clip(cy + dy, 0, h)

    new_circles[modified] = (new_cx, new_cy, r)

    return new_circles, modified, dx, dy



def simulated_annealing_with_lgb(circles, lgb_model):
    temp = 10.0
    n = 50

    while covered_percen(circles) < 0.97:
        for _ in range(int(n)):
            candidates = []
            features_list = []

            for _ in range(5):
                new_circles, idx, dx, dy = modify_circles_with_info(circles)

                features = extract_features(circles, new_circles, idx, dx, dy)
                features_list.append(features)
                candidates.append((new_circles, idx, dx, dy))

            features_array = np.array(features_list)

            pred_delta = lgb_model.predict(features_array)[0]

            best_idx = np.argmin(pred_delta)
            best_move = candidates[best_idx]

            old_cost = total_cost(circles)
            new_cost = total_cost(best_move[0])
            delta_f = new_cost - old_cost

            if delta_f > 0:
                acceptance = np.exp(-delta_f / temp)
                if random.random() < acceptance:
                    circles = best_move[0]
            else:
                circles = best_move[0]

            
        temp *= 0.995
        

    return circles
