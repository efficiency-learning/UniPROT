
import numpy as np
def greedy_selection(scores, interaction_matrix, K):
    """
    Select K data points based on the highest scores, dynamically updating scores
    by subtracting interactions with previously selected data points.

    Parameters:
    - scores: A numpy array of initial scores for each data point. (train_bs, val_bs)
    - interaction_matrix: A numpy matrix of pairwise interactions between data points. (train_bs, train_bs)
    - K: The number of data points to select.

    Returns:
    - selected_indices: Indices of the selected data points.
    """
    # Ensure scores is a mutable numpy array to update it in-place
    scores = scores.copy()
    interaction_matrix = interaction_matrix.copy()
    selected_indices = []
    K = min(K, scores.shape[0])
    print("scores", scores.shape, interaction_matrix.shape, K)

    for _ in range(K):
        # Select the index with the highest score
        idx_max = np.argmax(scores)
        selected_indices.append(idx_max)

        # Update scores by subtracting interactions with the selected data point
        scores -= interaction_matrix[idx_max, :]

        # Set the score of the selected data point to a very large negative value
        # to ensure it's not selected again
        scores[idx_max] = -np.inf

    return selected_indices