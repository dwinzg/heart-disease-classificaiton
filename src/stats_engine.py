import numpy as np
import scipy.stats
from sklearn.metrics import roc_curve, auc

def calculate_auc_ci(y_true, y_pred, n_bootstraps=5000):
    rng = np.random.RandomState(100)
    bootstrapped_scores = []
    y_true_arr = np.array(y_true)
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true_arr[indices])) < 2:
            continue
        score = auc(*roc_curve(y_true_arr[indices], y_pred[indices])[:2])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return ci_lower, ci_upper

# DeLong test 
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    tx = np.empty([positive_examples.shape[0], m], dtype=float)
    ty = np.empty([negative_examples.shape[0], n], dtype=float)
    tz = np.empty([predictions_sorted_transposed.shape[0], m + n], dtype=float)
    for r in range(predictions_sorted_transposed.shape[0]):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return 10 ** (scipy.stats.norm.logsf(z) / np.log(10)) * 2

def delong_roc_test(y_true, y_pred1, y_pred2):
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)
    
    # Sort predictions so that the positive class labels come first
    order = np.argsort(y_true)[::-1]
    y_true_sorted = y_true[order]
    
    predictions_sorted_transposed = np.vstack((y_pred1[order], y_pred2[order]))
    label_1_count = np.sum(y_true_sorted)
    
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    p_value = calc_pvalue(aucs, delongcov)
    return p_value[0][0]