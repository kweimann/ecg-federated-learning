from typing import Sequence, Tuple

import numpy as np
import torch
from scipy.signal import resample
from torch import Tensor
from wfdb import Record


class Resample:
  def __init__(self, fs: int):
    assert fs > 0
    self.fs = fs

  def __call__(self, record: Record) -> Record:
    if record.fs != self.fs:
      size, _ = record.p_signal.shape
      resampled_length = int(size * self.fs / record.fs)
      p_signal = resample(record.p_signal, num=resampled_length)
      return Record(
        p_signal=p_signal,
        record_name=record.record_name,
        fs=self.fs,
        sig_name=record.sig_name,
        comments=record.comments)
    return record


class CenterCrop:
  def __init__(self, size: int):
    assert size > 0
    self.size = size

  def __call__(self, record: Record) -> Record:
    size, _ = record.p_signal.shape
    if size > self.size:
      offset = (size - self.size) // 2
      p_signal = record.p_signal[offset:offset + self.size]
      return Record(
        p_signal=p_signal,
        record_name=record.record_name,
        fs=record.fs,
        sig_name=record.sig_name,
        comments=record.comments)
    return record


class MaskedPad:
  def __init__(self, size):
    assert size > 0
    self.size = size

  def __call__(self, record: Record) -> Tuple[Record, Tensor]:
    size, num_channels = record.p_signal.shape
    if size < self.size:
      sig_start = self.size - size
      p_signal = np.zeros_like(record.p_signal, shape=(self.size, num_channels))
      p_signal[sig_start:] = record.p_signal
      mask = torch.zeros(self.size)
      mask[sig_start:] = 1.
    else:
      p_signal = record.p_signal
      mask = torch.ones(size)
    record = Record(
      p_signal=p_signal,
      record_name=record.record_name,
      fs=record.fs,
      sig_name=record.sig_name,
      comments=record.comments)
    return record, mask


class Normalize:
  def __call__(self, record: Record) -> Record:
    mean = record.p_signal.mean(axis=0)
    std = record.p_signal.std(axis=0)
    p_signal = (record.p_signal - mean) / (std + 1e-9)
    return Record(
      p_signal=p_signal,
      record_name=record.record_name,
      fs=record.fs,
      sig_name=record.sig_name,
      comments=record.comments)


class SelectLeads:
  def __init__(self, leads: Sequence[str]):
    assert len(set(leads)) == len(leads)  # all leads are unique
    self.leads = leads

  def __call__(self, record: Record) -> Record:
    lead_indices = []
    for selected_lead in self.leads:
      lead_indices.append(record.sig_name.index(selected_lead))
    p_signal = record.p_signal[:, lead_indices]
    sig_name = list(self.leads)
    return Record(
      p_signal=p_signal,
      record_name=record.record_name,
      fs=record.fs,
      sig_name=sig_name,
      comments=record.comments)


class ReplaceNaN:
  """Replace nan with the mean computed over an entire lead."""
  def __call__(self, record: Record) -> Record:
    if np.isnan(record.p_signal).any():
      _, num_channels = record.p_signal.shape
      mean = np.nanmean(record.p_signal, axis=0)
      p_signal = record.p_signal.copy()
      for i in range(num_channels):
        np.nan_to_num(p_signal[:, i], copy=False, nan=mean[i])
      return Record(
        p_signal=p_signal,
        record_name=record.record_name,
        fs=record.fs,
        sig_name=record.sig_name,
        comments=record.comments)
    return record


class ToTensor:
  def __init__(self, channels_first: bool = False):
    self.channels_first = channels_first

  def __call__(self, record: Record) -> Tensor:
    p_signal = torch.from_numpy(record.p_signal).float()
    if self.channels_first:
      p_signal = p_signal.transpose(0, 1)
    return p_signal
