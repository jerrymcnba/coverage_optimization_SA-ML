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