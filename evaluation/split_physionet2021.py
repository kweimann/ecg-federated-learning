import argparse
import logging.config
from os import path

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm

from datasets import physionet2021

parser = argparse.ArgumentParser()
parser.add_argument('--db-path', required=True, help='database path')
parser.add_argument('--test-size', default=0.2, type=float, help='test size')
parser.add_argument('--seed', type=int, help='random state')
args = parser.parse_args()

logging.config.fileConfig('logging.ini')

if args.seed is not None:
  logging.info(f'setting seed {args.seed}')
  np.random.seed(args.seed)

record_names = []
diagnoses = []

for record_name in tqdm(physionet2021.list_records(args.db_path), desc='reading headers'):
  record_path = path.join(args.db_path, record_name)
  header = wfdb.rdheader(record_path)
  record_names.append(record_name)
  diagnoses.append(physionet2021.get_diagnoses(header))

mlb = MultiLabelBinarizer()
X = np.array(record_names)[:, np.newaxis]  # `iterative_train_test_split` expects 2D array
y = mlb.fit_transform(diagnoses)

unique_diagnoses = np.unique(y, axis=0)
logging.info(f'data: # records {len(X)} # unique diagnoses {len(unique_diagnoses)}')

shuffled_idx = np.random.choice(len(X), size=len(X), replace=False)

X_train, y_train, X_test, y_test = iterative_train_test_split(
  X[shuffled_idx], y[shuffled_idx], test_size=args.test_size)

unique_train_diagnoses = np.unique(y_train, axis=0)
logging.info(f'train: # records {len(X_train)} # unique diagnoses {len(unique_train_diagnoses)}')

unique_test_diagnoses = np.unique(y_test, axis=0)
logging.info(f'test: # records {len(X_test)} # unique diagnoses {len(unique_test_diagnoses)}')

train_df = pd.DataFrame(data={'record': X_train.squeeze(axis=1),
                              'train': True,
                              'test': False}).set_index('record')

test_df = pd.DataFrame(data={'record': X_test.squeeze(axis=1),
                             'train': False,
                             'test': True}).set_index('record')

data_split_df = pd.concat([train_df, test_df])
assert data_split_df.index.is_unique  # make sure a record cannot belong to both train and test
data_split_df = data_split_df.sort_index()

db_name = path.basename(args.db_path)
data_split_file = path.join('evaluation', f'split_{db_name}.csv')
logging.info(f"saving split in '{data_split_file}'")

data_split_df.to_csv(data_split_file)
