import numpy as np

def load_data(filename):
        return np.loadtxt(filename)

# simple test to make sure the euclidean distance function works

#a = np.array([1, 2, 3])
#b = np.array([4, 5, 6])
#print (euclidean_distance(a, b))  # should print 5.196152422706632

#nearest neighbor classifier

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def nearest_neighbor(data, feature_indices):
    N = int(data.shape[0])
    labels = data[:, 0]
    X = data[:, feature_indices]
    correct = 0

    for i in range(N):
        best_distance = float('inf')
        best_label = None
        for j in range(N):
            if i == j:
                continue
            distance = euclidean_distance(X[i], X[j])
            if distance < best_distance:
                best_distance = distance
                best_label = labels[j]
        if best_label == labels[i]:
            correct += 1
    return correct / N

#testing if the nearest neighbor classifier works
if __name__ == "__main__":
    data = load_data('CS170_Small_DataSet__94.txt')
    # using columns 1..3 as features (column 0 is the label)
    all_features = list(range(1, data.shape[1]))
    print(nearest_neighbor(data, all_features))