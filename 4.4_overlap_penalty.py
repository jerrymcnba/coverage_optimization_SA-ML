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