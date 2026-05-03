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