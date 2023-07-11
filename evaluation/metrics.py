import numpy as np
from scipy.special import expit
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.utils.extmath import stable_cumsum


def get_f1_scores(logits: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
  _, num_diagnoses = targets.shape
  probabilities = expit(logits)  # sigmoid activation
  available_diagnoses = thresholds < 1
  f1_scores = np.zeros(num_diagnoses)
  f1_scores[available_diagnoses] = f1_score(
    targets[:, available_diagnoses],
    probabilities[:, available_diagnoses] >= thresholds[np.newaxis, available_diagnoses],
    average=None, zero_division=0)
  return f1_scores


def optimize_thresholds(logits: np.ndarray, targets: np.ndarray):
  """Optimize prediction thresholds for f1 score."""
  _, num_diagnoses = targets.shape
  probabilities = expit(logits)  # sigmoid activation
  dx_idx, = targets.sum(axis=0).nonzero()  # optimize thresholds for existing diagnoses only
  f1_scores = np.zeros(num_diagnoses)
  best_thresholds = np.ones(num_diagnoses)  # default value of 1 means there is no threshold
  for dx in dx_idx:
    precision, recall, thresholds = precision_recall_curve(
      targets[:, dx], probabilities[:, dx])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_f1_idx = f1.argmax()
    f1_scores[dx] = f1[best_f1_idx]
    best_thresholds[dx] = thresholds[best_f1_idx]
  return f1_scores, best_thresholds


def optimize_local_thresholds(logits: np.ndarray, targets: np.ndarray):
  probabilities = expit(logits)  # sigmoid activation
  dx_idx, = targets.sum(axis=0).nonzero()
  local_scores = {}
  for dx in dx_idx:
    fps, tps, thresholds = _binary_clf_curve(targets[:, dx], probabilities[:, dx])
    local_scores[dx] = (fps, tps, thresholds)
  return local_scores


def optimize_central_thresholds(local_scores_list, num_diagnoses):
  # Group clients' scores by diagnosis.
  grouped_scores = {}
  for local_scores in local_scores_list:
    for dx, dx_scores in local_scores.items():
      grouped_scores.setdefault(dx, []).append(dx_scores)
  # merge clients' scores
  f1_scores = np.zeros(num_diagnoses)
  best_thresholds = np.ones(num_diagnoses)
  for dx, dx_scores_list in grouped_scores.items():
    fps, tps, merged_thresholds = _merge_binary_clf_curves(*zip(*dx_scores_list))
    precision, recall, thresholds = _precision_recall_curve(fps, tps, merged_thresholds)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_f1_idx = f1.argmax()
    best_thresholds[dx] = thresholds[best_f1_idx]
    f1_scores[dx] = f1[best_f1_idx]
  return f1_scores, best_thresholds


def _precision_recall_curve(fps, tps, thresholds):
  """simplified copy of sklearn.metrics.precision_recall_curve"""
  ps = tps + fps
  precision = np.divide(tps, ps, where=(ps != 0))

  if tps[-1] == 0:
    recall = np.ones_like(tps)
  else:
    recall = tps / tps[-1]

  precision = np.hstack((precision[::-1], 1))
  recall = np.hstack((recall[::-1], 0))
  thresholds = thresholds[::-1]

  return precision, recall, thresholds


def _binary_clf_curve(targets, probabilities):
  """simplified copy of sklearn.metrics._binary_clf_curve"""
  def is_1d(array): return array.ndim == 1
  def is_binary(array): return np.array_equal(array, array.astype(bool))
  def is_probability(array): return np.all((0 <= array) & (array <= 1))

  assert len(targets) == len(probabilities)
  assert is_1d(targets) and is_binary(targets)
  assert is_1d(probabilities) and is_probability(probabilities)

  desc_prob_indices = np.argsort(probabilities, kind="mergesort")[::-1]
  targets = targets[desc_prob_indices]
  probabilities = probabilities[desc_prob_indices]

  distinct_prob_indices = np.where(np.diff(probabilities))[0]
  threshold_indices = np.r_[distinct_prob_indices, targets.size - 1]

  tps = stable_cumsum(targets)[threshold_indices]
  fps = 1 + threshold_indices - tps
  thresholds = probabilities[threshold_indices]

  return fps, tps, thresholds


def _merge_binary_clf_curves(fps_list, tps_list, thresholds_list):
  all_thresholds = np.unique([threshold for split in thresholds_list for threshold in split])[::-1]
  fps_sum = np.zeros_like(all_thresholds)
  tps_sum = np.zeros_like(all_thresholds)

  for fps, tps, thresholds in zip(fps_list, tps_list, thresholds_list):
    idx = np.searchsorted(-all_thresholds, -thresholds)  # search in descending order
    assert np.array_equal(all_thresholds[idx], thresholds)

    for i, start in enumerate(idx):
      end = idx[i + 1] if i + 1 < len(thresholds) else None
      fps_sum[start:end] += fps[i]
      tps_sum[start:end] += tps[i]

  return fps_sum, tps_sum, all_thresholds


if __name__ == '__main__':
  np.random.seed(1236)

  num_samples = 1000
  num_classes = 10
  class_prob = [np.random.random() for _ in range(num_classes)]

  y_true = np.empty((num_samples, num_classes))

  for i in range(num_classes):
    y_true[:, i] = np.random.binomial(1, class_prob[i], (num_samples,))

  y_prob = np.random.random((num_samples, num_classes))

  num_splits = 5
  splits = np.sort(np.random.choice(num_samples - 1, size=num_splits - 1, replace=False) + 1)
  splits = np.concatenate([np.array([0]), splits])

  for i in range(num_classes):
    clf_curves = []
    for k in range(num_splits):
      start = splits[k]
      end = splits[k + 1] if k + 1 < num_splits else None
      clf_curves.append(_binary_clf_curve(y_true[start:end, i], y_prob[start:end, i]))

    merged_fps, merged_tps, merged_thresholds = _merge_binary_clf_curves(*zip(*clf_curves))
    fps, tps, thresholds = _binary_clf_curve(y_true[:, i], y_prob[:, i])

    assert np.array_equal(fps, merged_fps)
    assert np.array_equal(tps, merged_tps)
    assert np.array_equal(thresholds, merged_thresholds)

    precision, recall, thresholds = _precision_recall_curve(merged_fps, merged_tps, merged_thresholds)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    best = f1.argmax()
    min_viable_threshold = thresholds[best - 1] + 1e-9 if best > 0 else 0
    print(f'[{i + 1}] p={class_prob[i]:.5f} f1={f1[best]:.5f} | '
          # we can pick any threshold in the interval (i-1,i] and the score will not change
          #  based on that knowledge, we can merge threshold intervals and corresponding scores
          f'{f1_score(y_true[:, i], y_prob[:, i] >= min_viable_threshold):.5f} '
          f'threshold={thresholds[best]:.5f} (min viable={min_viable_threshold:.5f})')
