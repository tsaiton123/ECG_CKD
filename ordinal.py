"""
Ordinal Entropy regularizer
"""
import tensorflow as tf
import numpy as np
import random

def ordinal_entropy(features, gt):
    """
    Features: The last layer's features
    gt: The corresponding ground truth values
    """

    """
    sample in case the training size too large
    """
    # samples = random.sample(range(0, len(gt)-1), 100)  # random sample 100 features
    # samples = random.sample(range(0, len(gt)-1), 2)  # random sample 100 features
    # features = features[samples]
    # gt = gt[samples]

    """
    calculate distances in the feature space, i.e. ||z_{c_i} - z_{c_j}||_2

    """
    if not isinstance(features, tf.Tensor):
      features = tf.convert_to_tensor(features)
    if not isinstance(gt, tf.Tensor):
      gt = tf.convert_to_tensor(gt)
    p = tf.nn.l2_normalize(features, axis=1)
    _distance = euclidean_dist(p, p)
    _distance = up_triu(_distance)

    """
    calculate the distances in the label space, i.e. w_{ij} = ||y_i -y_j||_2
    """
    _weight = euclidean_dist(gt, gt)
    _weight = up_triu(_weight)
    _max = tf.reduce_max(_weight)
    _min = tf.reduce_min(_weight)
    _weight = ((_weight - _min) / _max)

    """
    L_d = - mean(w_ij ||z_{c_i} - z_{c_j}||_2)
    """
    _distance = _distance * _weight
    L_d = -tf.reduce_mean(_distance)

    return L_d


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = tf.shape(x)[0], tf.shape(y)[0]
    xx = tf.reduce_sum(tf.pow(x, 2), axis=1, keepdims=True)
    yy = tf.reduce_sum(tf.pow(y, 2), axis=1, keepdims=True)
    yy = tf.transpose(yy)

    dist = xx + yy
    xy = tf.matmul(x, y, transpose_b=True)
    dist -= 2 * xy
    dist = tf.sqrt(tf.maximum(dist, tf.constant(1e-12)))  # for numerical stability
    return dist

def up_triu(x):
    n = tf.shape(x)[0]
    triu_indices = np.triu_indices(n, k=1)
    return tf.gather_nd(x, tf.stack(triu_indices, axis=1))


if __name__ == '__main__':
  print(1)