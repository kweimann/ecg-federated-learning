import collections
import logging

import numpy as np


class Meter:
  def __init__(self, name, initial=0., fmt=':.4f'):
    self.name = name
    self.fmt = fmt
    self.value = initial

  def update(self, *args, **kwargs) -> None:
    raise NotImplementedError

  def __repr__(self):
    fmt = '{name} {value' + self.fmt + '}'
    return fmt.format(name=self.name, value=self.value)


class SimpleMeter(Meter):
  def update(self, value: any) -> None:
    self.value = value


class AverageMeter(Meter):
  def __init__(self, name, initial=0., fmt=':.4f'):
    super().__init__(name, initial=initial, fmt=fmt)
    self.running_value = 0.
    self.count = 0

  def update(self, value: float, count: int = 1) -> None:
    self.running_value += value
    self.count += count
    self.value = self.running_value / self.count


class LossMeter(AverageMeter):
  def __init__(self, name, initial=0., fmt=':.4e'):
    super().__init__(name, initial=initial, fmt=fmt)


class Monitor:
  def __init__(self, meter: Meter, cmp=np.greater):
    self.meter = meter
    self.cmp = cmp
    self._best_value = None

  def update(self, *args, **kwargs) -> bool:
    self.meter.update(*args, **kwargs)
    if self._best_value is None or self.cmp(self.meter.value, self._best_value):
      self._best_value = self.meter.value
      return True
    return False

  def __repr__(self):
    return repr(self.meter)


class Summary:
  def __init__(self, ids):
    self._values = collections.OrderedDict(
      (summary_id, []) for summary_id in ids)

  def update(self, values: dict = None, **kwargs):
    if values is None:
      values = kwargs
    else:
      values = {**values, **kwargs}  # note: kwargs overrides keys in the summary dict
    assert sorted(values) == sorted(self._values), 'values are not consistent with the summary ids'
    for summary_id, value in values.items():
      if isinstance(value, Meter):
        value = np.array(value.value)
      self._values[summary_id].append(value)

  def collect(self):
    return collections.OrderedDict(
      {summary_id: np.stack(values)
       for summary_id, values in self._values.items()})


class LoggingFormatter(logging.Formatter):
  def format(self, record):
    elapsed = int(record.relativeCreated)
    milliseconds = elapsed % 1000
    seconds = elapsed // 1000 % 60
    minutes = elapsed // (1000 * 60) % 60
    hours = elapsed // (1000 * 60 * 60)
    record.elapsed = f'{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}'
    return super().format(record)
