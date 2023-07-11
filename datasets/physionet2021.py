import logging
import re
from os import path, listdir
from typing import List, Sequence, Iterator

import numpy as np
import pandas as pd
import torch.utils.data
import wfdb
from pandas import DataFrame
from torch import Tensor
from wfdb import Record

LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')

DB_COLS = {
  'CPSC': 'WFDB_CPSC2018',
  'CPSC_Extra': 'WFDB_CPSC2018_2',
  'PTB': 'WFDB_PTB',
  'PTB_XL': 'WFDB_PTBXL',
  'Georgia': 'WFDB_Ga',
  'Chapman_Shaoxing': 'WFDB_ChapmanShaoxing',
  'Ningbo': 'WFDB_Ningbo'
}

CLIENTS = {
  'ChapmanShaoxing': ['WFDB_ChapmanShaoxing'],
  'CPSC2018': ['WFDB_CPSC2018', 'WFDB_CPSC2018_2'],
  'Georgia': ['WFDB_Ga'],
  'Ningbo': ['WFDB_Ningbo'],
  'PTB': ['WFDB_PTB', 'WFDB_PTBXL']
}


class ECGDataset(torch.utils.data.Dataset):
  """In-memory datasets of ECG records: signals and diagnoses."""
  def __init__(self, records: Sequence[Record], dx_mapping: DataFrame, transform=None):
    self.records = records
    self.dx_mapping = dx_mapping
    self.transform = transform
    diagnosis_codes = list(set(diagnosis for record in records for diagnosis in get_diagnoses(record)))
    # Mask indicates which diagnoses are available in this dataset.
    #  Mask should be created from all records in the database!
    #  Dataset can be split into train and test with torch.utils.data.Subset
    self._target_mask = _one_hot_encode(dx_mapping, diagnosis_codes)

  def __len__(self) -> int:
    return len(self.records)

  def __getitem__(self, index: int):
    record = self.records[index]
    target = self._encode_diagnoses(record)
    if self.transform is not None:
      record = self.transform(record)
    return record, (target, self._target_mask)

  def _encode_diagnoses(self, record: Record) -> Tensor:
    return _one_hot_encode(self.dx_mapping, get_diagnoses(record))

  @classmethod
  def from_path(cls, db_path: str, dx_mapping: DataFrame, transform=None):
    return cls(records=list(stream_records(db_path)),
               dx_mapping=dx_mapping, transform=transform)


def get_diagnoses(record: Record) -> List[int]:
  comments = dict(field.split(': ') for field in record.comments)
  return [int(diagnosis) for diagnosis in comments['Dx'].split(',')]


def read_dx_mapping(evaluation_path: str) -> DataFrame:
  scored_path = path.join(evaluation_path, 'dx_mapping_scored.csv')
  unscored_path = path.join(evaluation_path, 'dx_mapping_unscored.csv')
  scored_df = pd.read_csv(scored_path, index_col='SNOMEDCTCode')
  unscored_df = pd.read_csv(unscored_path, index_col='SNOMEDCTCode')
  scored_df['scored'] = True
  unscored_df['scored'] = False
  dx_mapping = pd.concat([scored_df, unscored_df], sort=False)
  dx_mapping = dx_mapping.sort_values(by='Abbreviation')
  dx_mapping['idx'] = range(len(dx_mapping))
  return dx_mapping


def remap_same_diagnoses(dx_mapping: DataFrame) -> DataFrame:
  """If according to the notes, two diagnoses are scored the same,
  this function remaps their indices (`idx`) to a single value."""
  idx = []
  for code, row in dx_mapping.iterrows():
    notes = row['Notes']
    if not pd.isna(notes):
      remaining_notes = ''
      matched_codes = []
      i = 0
      for match in re.finditer('\\d+', notes):  # find codes (integers)
        matched_codes.append(int(match.group()))
        start, end = match.span()
        remaining_notes += notes[i:start]
        i = end
      remaining_notes += notes[i:]
      remaining_notes = ' '.join(remaining_notes.split())
      assert len(matched_codes) == 2
      assert code in matched_codes
      assert remaining_notes == 'We score and as the same diagnosis' \
             or remaining_notes == 'We score and as the same diagnosis.'
      left_code, right_code = matched_codes
      if left_code == code:
        left_code_idx = row['idx']
        idx.append(left_code_idx)  # use left_code as the code for the group
      elif right_code == code:
        left_code_idx = dx_mapping.loc[left_code]['idx']
        idx.append(left_code_idx)  # remap right_code to left_code
    else:
      idx.append(row['idx'])
  new_dx_mapping = dx_mapping.copy()
  new_dx_mapping['idx'] = idx
  return new_dx_mapping


def stream_records(db_path: str) -> Iterator[Record]:
  for record_name in list_records(db_path):
    record_path = path.join(db_path, record_name)
    record = wfdb.rdrecord(record_path)
    if record.record_name != record_name:  # fixes misspelled record name in the Ningbo database
      logging.warning(f'record name ({record.record_name}) is inconsistent '
                      f'with the filename ({record_name})')
      record.record_name = record_name
    record.p_signal = record.p_signal.astype(np.float32)
    yield record


def list_records(db_path: str) -> List[str]:
  files = [path.splitext(filename) for filename in listdir(db_path)]
  return sorted(record_name for record_name, ext in files if ext == '.hea')


def _one_hot_encode(dx_mapping: DataFrame, positive_index: Sequence[int]) -> Tensor:
  def series_to_tensor(series): return torch.from_numpy(series.to_numpy())
  class_vector = torch.zeros(len(dx_mapping))
  positive_index = series_to_tensor(
    dx_mapping.loc[positive_index]['idx']
  ).long()
  class_vector[positive_index] = 1
  return class_vector
