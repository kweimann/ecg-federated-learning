import gzip
import logging
import pickle
from os import path, makedirs

import torch
from tqdm import tqdm

from datasets import physionet2021


def read_data_check_cache(db_path, data_reader, cache_dir=None):
  db_name, *extension = path.basename(db_path).split('.')
  cached_file = f'{path.join(cache_dir, db_name)}.pkl.gz'
  extension = '.'.join(extension)
  if path.isfile(db_path) and extension == 'pkl.gz':
    logging.debug(f'reading cached records from {db_path}')
    data = load_pkl(db_path)
  elif path.isfile(cached_file):
    logging.debug(f'reading cached records from {cached_file}')
    data = load_pkl(cached_file)
  elif path.isdir(db_path):
    logging.debug(f'reading records from {db_path}')
    data = data_reader(db_path)
    if cache_dir is not None:
      logging.debug(f'caching records in {cached_file}')
      makedirs(cache_dir, exist_ok=True)
      save_pkl(data, cached_file)
  else:
    raise ValueError(f'Cannot read records from {db_path}')
  return data


def physionet2021_reader(dx_mapping, transform=None, debug=False):
  def data_reader(db_path):
    records_stream = physionet2021.stream_records(db_path)
    if debug:
      num_records = len(physionet2021.list_records(db_path))
      records_stream = tqdm(records_stream, desc='reading records', total=num_records)
    dataset = physionet2021.ECGDataset(list(records_stream), dx_mapping, transform=transform)
    record_names = [record.record_name for record in dataset.records]
    if debug:
      dataset = tqdm(dataset, desc='pre-processing records')
    inputs, targets = zip(*dataset)
    x, input_mask = map(torch.stack, zip(*inputs))
    targets, target_mask = map(torch.stack, zip(*targets))
    return (x, input_mask, target_mask), targets, record_names
  return data_reader


def load_pkl(file, compress='infer'):
  """Load an object from a pickled file. If `compress` is set to `infer`,
  decide whether the file is compressed based on its extension."""
  if compress == 'infer':
    _, ext = path.splitext(file)
    compress = ext == '.gz'
  if compress:
    with gzip.open(file, 'rb') as fh:
      return pickle.load(fh)
  else:
    with open(file, 'rb') as fh:
      return pickle.load(fh)


def save_pkl(obj, file, compress='infer'):
  """Save an object in a pickle file. If `compress` is set to `infer`,
  decide whether to compress the file based on its extension."""
  if compress == 'infer':
    _, ext = path.splitext(file)
    compress = ext == '.gz'
  if compress:
    with gzip.open(file, 'wb') as fh:
      pickle.dump(obj, fh, protocol=4)
  else:
    with open(file, 'wb') as fh:
      pickle.dump(obj, fh, protocol=4)
