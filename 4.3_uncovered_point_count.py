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