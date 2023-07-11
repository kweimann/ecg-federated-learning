import argparse
import copy
import functools
import itertools
import logging.config
import math
import random
from dataclasses import dataclass
from os import path, makedirs
from time import time
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch.utils.data
from skmultilearn.model_selection import iterative_train_test_split
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from wfdb import Record

import datasets
import evaluation
import models
import utils
from datasets import physionet2021, transforms

PHYSIONET2021_DATABASES = {
  database: client
  for client, databases in physionet2021.CLIENTS.items()
  for database in databases
}

parser = argparse.ArgumentParser()
parser.add_argument('--job-dir', default='.', help='job directory')
parser.add_argument('--cache-dir', help='cache directory')
parser.add_argument('--train-db', nargs='+', default=[], help='path to databases used for training')
parser.add_argument('--test-db', nargs='+', default=[], help='path to databases used for testing')
parser.add_argument('--no-split-test', action='store_true', help='entire test-db is used for testing')
parser.add_argument('--val', default=0.05, type=float, help='percentage of data used for validation')
parser.add_argument('--checkpoint', help='path to the checkpoint file')
parser.add_argument('--optimization', default='central', choices=['central', 'FedOpt', 'MR-MTL'],
                    help='type of optimization')
parser.add_argument('--fl-adaptive-opt', action='store_true', help='adaptive optimization in federated learning')
parser.add_argument('--server-lr', default=0.001, type=float, help='learning rate on the server')
parser.add_argument('--server-momentum', default=0., type=float, help='SGD momentum on the server')
parser.add_argument('--server-beta1', default=0.9, type=float, help='beta1 on the server if using FedAdam ')
parser.add_argument('--server-beta2', default=0.999, type=float, help='beta2 on the server if using FedAdam')
parser.add_argument('--remove-bias-correction', action='store_true', help='no bias correction in FedAdam')
parser.add_argument('--cosine-lr-schedule', action='store_true', help='cosine learning rate schedule')
parser.add_argument('--warmup-steps', default=30, type=int,
                    help='number of warmup steps before cosine learning rate schedule')
parser.add_argument('--client-lr', default=0.1, type=float, help='learning rate on the clients')
parser.add_argument('--client-momentum', default=0., type=float, help='SGD momentum on the clients')
parser.add_argument('--batch-size', default=64, type=int, help='batch size during training and evaluation')
parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
parser.add_argument('--steps-per-epoch', default=None, type=int, help='limit on the number of steps per epoch')
parser.add_argument('--fs', default=128, type=int, help='sampling frequency (Hz)')
parser.add_argument('--duration', default=60, type=int, help='signal duration (seconds)')
parser.add_argument('--arch', default='resnet34', choices=['resnet18', 'resnet34'], help='ResNet architecture')
parser.add_argument('--attention-pooling', action='store_true',
                    help='attention pooling instead of global average pooling')
parser.add_argument('--uniform-client-weights', action='store_true', help='same weight for all clients')
parser.add_argument('--mr-mtl-kappa', default=1.0, type=float, help='MR-MTL regularization strength')
parser.add_argument('--seed', type=int, help='random state')


def main(args):
  is_training = len(args.train_db) > 0 and args.epochs > 0
  is_testing = len(args.test_db) > 0

  if not is_training and not is_testing:
    logging.warning('Both training and testing modes are off')
    return

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  logging.debug(f'Using {device}')

  if args.seed is not None:
    logging.debug(f'Setting random state to {args.seed}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

  makedirs(args.job_dir, exist_ok=True)

  # Read diagnosis mapping.
  dx_mapping = physionet2021.read_dx_mapping('evaluation')
  dx_mapping = physionet2021.remap_same_diagnoses(dx_mapping)
  scored_diagnoses = series_to_tensor(dx_mapping['scored']).bool()
  num_diagnoses = len(dx_mapping)

  # Initialize data reader.
  physionet2021_data_reader = datasets.physionet2021_reader(
    dx_mapping=dx_mapping,
    transform=RecordPreprocessor(args.duration, args.fs),
    debug=False)

  physionet2021_data = {}

  # Initialize the model.
  if args.arch == 'resnet18':
    stages = [2, 2, 2, 2]
  elif args.arch == 'resnet34':
    stages = [3, 4, 6, 3]
  else:
    raise ValueError(f'Unsupported ResNet architecture: {args.arch}')

  def make_model():
    return models.AttentiveResNet(
      use_attn_pool=args.attention_pooling,
      sig_len=args.fs * args.duration,
      stages=stages,
      num_outputs=num_diagnoses,
      kernel_size=[7, 5, 5, 3],
      stem_kernel_size=7
    ).to(device)

  central_model = make_model()

  logging.debug(f'#parameters {sum(p.numel() for p in central_model.parameters()):,d}')

  if args.checkpoint:
    logging.info(f'Loading checkpoint from {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint)
  else:
    checkpoint = None

  if is_training:
    if checkpoint is not None:
      logging.info('Loading model weights from the checkpoint.')
      missing_keys, unexpected_keys = central_model.load_state_dict(checkpoint['model'], strict=False)
      # note that we do not load the state of the optimizer (and training in general)
      if missing_keys:
        logging.warning(f'Missing keys: {", ".join(missing_keys)}')
      if unexpected_keys:
        logging.warning(f'Unexpected keys: {", ".join(unexpected_keys)}')

    # For each client, prepare training and validation sets.
    train_data = {}

    for train_db in args.train_db:
      db_name, *_ = path.basename(train_db).split('.')
      client_name = PHYSIONET2021_DATABASES[db_name]

      if train_db in physionet2021_data:
        logging.warning(f'Skipping {db_name} because it was already loaded.')
        continue

      logging.info(f'Loading data from {db_name}')

      data = datasets.read_data_check_cache(
        db_path=train_db,
        data_reader=physionet2021_data_reader,
        cache_dir=args.cache_dir)

      data_split = split_data(
        db_path=train_db,
        data=data,
        val=args.val)

      physionet2021_data[train_db] = data
      train_data.setdefault(client_name, []).append(data_split)

    client_names, client_databases = zip(*train_data.items())

    training_sets = [ConcatDataset(training_sets) for databases in client_databases
                     if (training_sets := [Subset(db.dataset, db.training_idx) for db in databases
                                           if db.training_idx is not None])]

    validation_sets = [ConcatDataset(validation_sets) for databases in client_databases
                       if (validation_sets := [Subset(db.dataset, db.validation_idx) for db in databases
                                               if db.validation_idx is not None])]

    assert len(training_sets) == len(validation_sets) > 0

    for client_name, train, val in zip(client_names, training_sets, validation_sets):
      logging.debug(f'{client_name}: #train {len(train)} #val {len(val)}')

    # For each client, compute available diagnoses by merging all target masks.
    available_diagnoses = [torch.cat([db.dataset.tensors[-1] for db in databases]).any(dim=0)
                           for databases in client_databases]

    # For each client, compute the intersection between the scored diagnoses
    #  and the diagnoses available for this client.
    evaluated_diagnoses = [available_diagnoses & scored_diagnoses
                           for available_diagnoses in available_diagnoses]

    all_evaluated_diagnoses = torch.stack(evaluated_diagnoses).any(dim=0)

    logging.debug(f'Evaluated diagnoses ({all_evaluated_diagnoses.sum()}): '
                  f'{", ".join(dx_mapping[all_evaluated_diagnoses.numpy()]["Abbreviation"])}')

    checkpoint_file = path.join(args.job_dir, 'checkpoint.pth')

    if args.optimization == 'central':
      optimizer = torch.optim.Adam(
        params=central_model.parameters(),
        lr=args.server_lr,
        betas=(args.server_beta1, args.server_beta2))

      train_loader = DataLoader(
        dataset=ConcatDataset(training_sets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1)

      val_loaders = [
        DataLoader(
          dataset=val,
          batch_size=args.batch_size,
          num_workers=1)
        for val in validation_sets
      ]

      val_client_f1_monitors = [utils.Monitor(utils.SimpleMeter('val_f1')) for _ in client_names]
      mean_val_f1_monitor = utils.Monitor(utils.SimpleMeter('val_f1'))
      summary = utils.Summary(['train_loss', 'val_loss', 'val_client_loss',
                               'train_f1', 'val_f1', 'val_client_f1'])

      for epoch in range(args.epochs):
        if args.cosine_lr_schedule:
          update_lr_with_cosine_schedule_(
            optimizer=optimizer,
            step=epoch,
            max_steps=args.epochs,
            max_lr=args.server_lr,
            min_lr=0.1 * args.server_lr,
            warmup_steps=args.warmup_steps
          )
        (train_logits, train_targets), (_, _, train_loss) = train_epoch(
          model=central_model,
          optimizer=optimizer,
          forward_pass_fn=forward_pass,
          dataloader=train_loader,
          clip_value=1.0,
          device=device)
        train_f1, thresholds = evaluation.optimize_thresholds(train_logits.numpy(), train_targets.numpy())
        mean_train_f1 = train_f1[all_evaluated_diagnoses].mean()
        # Evaluate the model separately for each client.
        val_client_outputs = [
          eval_epoch(
            model=central_model,
            forward_pass_fn=forward_pass,
            dataloader=val_loader,
            device=device)
          for val_loader in val_loaders]
        val_client_loss = [F.binary_cross_entropy_with_logits(logits, targets)
                           for logits, targets in val_client_outputs]
        val_client_f1 = [evaluation.get_f1_scores(logits.numpy(), targets.numpy(), thresholds)
                         for (logits, targets) in val_client_outputs]
        for client_name, monitor, f1_scores, target_mask in zip(
            client_names, val_client_f1_monitors, val_client_f1, evaluated_diagnoses):
          is_new_client_best = monitor.update(f1_scores[target_mask].mean())
          if is_new_client_best:
            torch.save({'model': central_model.state_dict(),
                        'thresholds': thresholds},
                       path.join(args.job_dir, f'checkpoint_{client_name}.pth'))
        # Evaluate the model jointly for all clients.
        val_logits, val_targets = map(torch.cat, zip(*val_client_outputs))
        val_loss = F.binary_cross_entropy_with_logits(val_logits, val_targets)
        val_f1 = evaluation.get_f1_scores(val_logits.numpy(), val_targets.numpy(), thresholds)
        is_new_best = mean_val_f1_monitor.update(val_f1[all_evaluated_diagnoses].mean())
        # Print the scores and save the model.
        logging.info(f'[{epoch + 1:03d}] {"(*)" if is_new_best else "   "} '
                     f'{train_loss} val_loss {val_loss:.4f} '
                     f'train_f1 {mean_train_f1:.4f} {mean_val_f1_monitor}')
        if is_new_best:
          torch.save({'model': central_model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'thresholds': thresholds},
                     checkpoint_file)
        summary.update(
          train_loss=train_loss,
          val_loss=val_loss,
          val_client_loss=val_client_loss,
          train_f1=train_f1,
          val_f1=val_f1,
          val_client_f1=val_client_f1)
        metrics = summary.collect()
        metrics['clients'] = np.array(client_names)
        metrics['diagnoses'] = dx_mapping['Abbreviation'].to_numpy()
        metrics['evaluated_diagnoses'] = torch.stack(evaluated_diagnoses).numpy()
        datasets.save_pkl(metrics, file=path.join(args.job_dir, 'train_metrics.pkl.gz'))
    else:  # Federated Learning
      if args.fl_adaptive_opt:
        logging.debug('Using adaptive optimizer.')
        optimizer = torch.optim.Adam(
          params=central_model.parameters(),
          lr=args.server_lr,
          betas=(args.server_beta1, args.server_beta2))

        if args.remove_bias_correction:
          logging.debug('Removing bias correction.')
          remove_bias_correction_(optimizer)
      else:
        optimizer = torch.optim.SGD(
          params=central_model.parameters(),
          lr=args.server_lr,
          momentum=args.server_momentum)

      if args.optimization == 'FedOpt':
        clients = [
          FedOptClient(
            optimizer=lambda parameters: torch.optim.SGD(  # stateless optimizer
              params=parameters,
              lr=args.client_lr,
              momentum=args.client_momentum),
            train_loader=DataLoader(
              dataset=train,
              batch_size=args.batch_size,
              shuffle=True,
              num_workers=1),
            val_loader=DataLoader(
              dataset=val,
              batch_size=args.batch_size,
              num_workers=1),
            steps_per_epoch=args.steps_per_epoch,
            clip_value=1.0,
            device=device,
            name=client_name)
          for client_name, train, val in zip(client_names, training_sets, validation_sets)]
      elif args.optimization == 'MR-MTL':
        clients = [
          MRMTLClient(
            local_model=make_model(),
            optimizer=lambda parameters: torch.optim.SGD(  # stateless optimizer
              params=parameters,
              lr=args.client_lr,
              momentum=args.client_momentum),
            kappa=args.mr_mtl_kappa,
            train_loader=DataLoader(
              dataset=train,
              batch_size=args.batch_size,
              shuffle=True,
              num_workers=1),
            val_loader=DataLoader(
              dataset=val,
              batch_size=args.batch_size,
              num_workers=1),
            steps_per_epoch=args.steps_per_epoch,
            clip_value=1.0,
            device=device,
            name=client_name)
          for client_name, train, val in zip(client_names, training_sets, validation_sets)]
        # initialize mean model on the central server
        local_params_iterator = zip(*[client.local_model.parameters() for client in clients])
        with torch.no_grad():
          for central_param, local_params in zip(central_model.parameters(), local_params_iterator):
            central_param.data = torch.stack(local_params).mean(dim=0)
      else:
        raise ValueError(f'Unknown optimization type: {args.optimization}')

      if args.uniform_client_weights:
        client_weights = torch.ones(len(clients), device=device) / len(clients)
      else:
        # Weight clients by the train counts. Clients share how much data they have.
        client_weights = weight_clients(training_sets, device=device)

      for client_name, weight in zip(client_names, client_weights):
        logging.debug(f'{client_name}: weight={weight:.4f}')

      summary = utils.Summary(['train_client_loss', 'val_client_loss', 'train_f1',
                               'train_client_f1_local', 'val_f1', 'val_client_f1',
                               'val_client_f1_local', 'grad_magnitude', 'grad_variance'])

      client_val_f1_monitors = [utils.Monitor(utils.SimpleMeter('val_f1')) for _ in clients]
      mean_val_f1_monitor = utils.Monitor(utils.SimpleMeter('val_f1'))

      for comm_round in range(args.epochs):
        if args.cosine_lr_schedule:
          update_lr_with_cosine_schedule_(
            optimizer=optimizer,
            step=comm_round,
            max_steps=args.epochs,
            max_lr=args.server_lr,
            min_lr=0.1 * args.server_lr,
            warmup_steps=args.warmup_steps
          )
        # Server distributes the model; client trains locally.
        updates, train_outputs, meters = zip(*[client.train_epoch(central_model) for client in clients])
        _, _, train_client_loss = zip(*meters)
        train_client_loss = [meter.value for meter in train_client_loss]
        local_scores = [evaluation.optimize_local_thresholds(logits.numpy(), targets.numpy())
                        for logits, targets in train_outputs]
        train_client_f1_local, client_thresholds = zip(
          *[evaluation.optimize_thresholds(logits.numpy(), targets.numpy())
            for logits, targets in train_outputs])
        del train_outputs  # Ultimately, clients may choose to share only the updates.
        # Clients communicate the updates; server merges the updates.
        grad_magnitude, grad_variance = gradient_metrics(updates, client_weights)
        for update, client_weight in zip(updates, client_weights):
          for param, param_update in zip(central_model.parameters(), update):
            if param.grad is None:
              param.grad = -(client_weight * param_update)  # note the negation
            else:
              param.grad -= (client_weight * param_update)  # note the negation
        optimizer.step()
        optimizer.zero_grad()
        # Calculate training loss and f1 scores.
        mean_train_loss = sum(client_weight * loss for client_weight, loss in zip(client_weights, train_client_loss))
        train_f1, thresholds = evaluation.optimize_central_thresholds(local_scores, num_diagnoses)
        mean_train_f1 = train_f1[all_evaluated_diagnoses].mean()
        # Evaluate the model separately for each client.
        val_client_outputs = [client.eval_epoch(central_model) for client in clients]
        val_client_loss = [F.binary_cross_entropy_with_logits(logits, targets)
                           for logits, targets in val_client_outputs]
        val_client_f1 = [evaluation.get_f1_scores(logits.numpy(), targets.numpy(), thresholds)
                         for logits, targets in val_client_outputs]
        val_client_f1_local = [evaluation.get_f1_scores(logits.numpy(), targets.numpy(), thresholds)
                               for (logits, targets), thresholds in zip(val_client_outputs, client_thresholds)]
        for client, monitor, f1_scores, target_mask, thresholds_local in zip(
            clients, client_val_f1_monitors, val_client_f1, evaluated_diagnoses, client_thresholds):
          is_new_client_best = monitor.update(f1_scores[target_mask].mean())
          if is_new_client_best:
            # If a local model does not exist, default to the central model.
            local_model = getattr(client, 'local_model', central_model)
            torch.save({'model': local_model.state_dict(),
                        'thresholds': thresholds,
                        'thresholds_local': thresholds_local},
                       path.join(args.job_dir, f'checkpoint_{client.name}.pth'))
        # evaluate central model
        val_logits, val_targets = map(torch.cat, zip(*val_client_outputs))
        mean_val_loss = sum(client_weight * loss for client_weight, loss in zip(client_weights, val_client_loss))
        val_f1 = evaluation.get_f1_scores(val_logits.numpy(), val_targets.numpy(), thresholds)
        is_new_best = mean_val_f1_monitor.update(val_f1[all_evaluated_diagnoses].mean())
        if is_new_best:
          torch.save({'model': central_model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'thresholds': thresholds},
                     checkpoint_file)
        summary.update(
          train_client_loss=train_client_loss,
          val_client_loss=val_client_loss,
          train_f1=train_f1,
          train_client_f1_local=train_client_f1_local,
          val_f1=val_f1,
          val_client_f1=val_client_f1,
          val_client_f1_local=val_client_f1_local,
          grad_magnitude=grad_magnitude.item(),
          grad_variance=grad_variance.item())
        logging.info(f'[{comm_round + 1:03d}] {"(*)" if is_new_best else "   "} '
                     f'train_loss {mean_train_loss:.4f} val_loss {mean_val_loss:.4f} '
                     f'train_f1 {mean_train_f1:.4f} {mean_val_f1_monitor}')
        metrics = summary.collect()
        metrics['clients'] = np.array([client.name for client in clients])
        metrics['client_weights'] = client_weights.cpu().numpy()
        metrics['diagnoses'] = dx_mapping['Abbreviation'].to_numpy()
        metrics['evaluated_diagnoses'] = torch.stack(evaluated_diagnoses).numpy()
        datasets.save_pkl(metrics, file=path.join(args.job_dir, 'train_metrics.pkl.gz'))

    if is_testing:
      logging.info(f'Loading the best checkpoint from {checkpoint_file}')
      checkpoint = torch.load(checkpoint_file)

  if is_testing:
    if checkpoint is not None:
      logging.info('Loading model weights from the checkpoint.')
      central_model.load_state_dict(checkpoint['model'])
      thresholds = checkpoint['thresholds']
      thresholds_local = checkpoint.get('thresholds_local')
    else:
      logging.warning('Using a randomly initialized model. '
                      'All class thresholds are set to a default value of 0.5')
      thresholds = np.full(num_diagnoses, 0.5)
      thresholds_local = None

    test_data = {}

    for test_db in args.test_db:
      db_name, *_ = path.basename(test_db).split('.')
      client_name = PHYSIONET2021_DATABASES[db_name]

      if test_db in physionet2021_data:
        data = physionet2021_data[test_db]
      else:
        logging.info(f'Loading data from {db_name}')
        data = datasets.read_data_check_cache(
          db_path=test_db,
          data_reader=physionet2021_data_reader,
          cache_dir=args.cache_dir)

      if args.no_split_test:
        (x, input_mask, target_mask), targets, record_names = data
        data_split = DataSplit(
          db_name=db_name,
          dataset=TensorDataset(x, input_mask, target_mask, targets),
          record_names=np.array(record_names),
          test_idx=np.arange(len(x)))
      else:
        data_split = split_data(
          db_path=test_db,
          data=data)

      test_data.setdefault(client_name, []).append(data_split)

    client_names, client_databases = zip(*test_data.items())

    test_sets = [ConcatDataset(test_sets) for databases in client_databases
                 if (test_sets := [Subset(db.dataset, db.test_idx) for db in databases
                                   if db.test_idx is not None])]

    assert len(test_sets) > 0

    for client_name, test in zip(client_names, test_sets):
        logging.debug(f'{client_name}: #test {len(test)}')

    available_diagnoses = [torch.cat([db.dataset.tensors[-1] for db in databases]).any(dim=0)
                           for databases in client_databases]

    available_diagnoses_for_model = torch.tensor(thresholds < 1)

    evaluated_diagnoses = [available_diagnoses & scored_diagnoses & available_diagnoses_for_model
                           for available_diagnoses in available_diagnoses]

    all_evaluated_diagnoses = torch.stack(evaluated_diagnoses).any(dim=0)

    logging.debug(f'Evaluated diagnoses ({all_evaluated_diagnoses.sum()}): '
                  f'{", ".join(dx_mapping[all_evaluated_diagnoses.numpy()]["Abbreviation"])}')

    test_loaders = [
      DataLoader(
        dataset=test,
        batch_size=args.batch_size,
        num_workers=1)
      for test in test_sets
    ]

    test_client_outputs = [
      eval_epoch(
        model=central_model,
        forward_pass_fn=forward_pass,
        dataloader=test_loader,
        device=device)
      for test_loader in test_loaders]

    test_client_f1 = [evaluation.get_f1_scores(logits.numpy(), targets.numpy(), thresholds)
                      for logits, targets in test_client_outputs]
    test_logits, test_targets = map(torch.cat, zip(*test_client_outputs))
    test_f1 = evaluation.get_f1_scores(test_logits.numpy(), test_targets.numpy(), thresholds)
    mean_test_f1 = test_f1[all_evaluated_diagnoses].mean()

    test_metrics = {'test_f1': test_f1,
                    'test_client_f1': np.array(test_client_f1),
                    'clients': np.array(client_names),
                    'diagnoses': dx_mapping['Abbreviation'].to_numpy(),
                    'evaluated_diagnoses': torch.stack(evaluated_diagnoses).numpy()}

    if thresholds_local is not None:
      available_diagnoses_for_model_local = torch.tensor(thresholds_local < 1)

      evaluated_diagnoses_local = [available_diagnoses & scored_diagnoses & available_diagnoses_for_model_local
                                   for available_diagnoses in available_diagnoses]

      all_evaluated_diagnoses_local = torch.stack(evaluated_diagnoses_local).any(dim=0)

      test_client_f1_local = [evaluation.get_f1_scores(logits.numpy(), targets.numpy(), thresholds_local)
                              for logits, targets in test_client_outputs]
      test_f1_local = evaluation.get_f1_scores(test_logits.numpy(), test_targets.numpy(), thresholds_local)
      mean_test_f1_local = test_f1_local[all_evaluated_diagnoses_local].mean()
      logging.info(f'test_f1 {mean_test_f1:.4f} test_f1_local {mean_test_f1_local:.4f}')
      test_metrics['test_f1_local'] = test_f1_local
      test_metrics['test_client_f1_local'] = np.array(test_client_f1_local)
      test_metrics['evaluated_diagnoses_local'] = evaluated_diagnoses_local
    else:
      logging.info(mean_test_f1)

    datasets.save_pkl(test_metrics, file=path.join(args.job_dir, 'test_metrics.pkl.gz'))

    # save predictions for every test record
    predictions_df = pd.DataFrame(
      data=test_logits.numpy(),
      columns=dx_mapping['Abbreviation'].to_numpy())

    db_names = np.concatenate(
      [np.repeat(db.db_name, len(db.test_idx))
       for databases in client_databases
       for db in databases
       if db.test_idx is not None])

    record_names = np.concatenate(
      [db.record_names[db.test_idx]
       for databases in client_databases
       for db in databases
       if db.test_idx is not None])

    predictions_df.insert(loc=0, column='db_name', value=db_names)
    predictions_df.insert(loc=1, column='record_name', value=record_names)

    predictions_df.to_csv(path.join(args.job_dir, 'predictions.csv'), index=False)


class FedOptClient:
  # computes an update of the central model on local data
  def __init__(self, optimizer, train_loader, val_loader,
               steps_per_epoch=None, clip_value=None, device=None, name=None):
    if not callable(optimizer) and not isinstance(optimizer, torch.optim.Optimizer):
      raise ValueError(f'Unknown optimizer: {optimizer}')
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.steps_per_epoch = steps_per_epoch
    self.clip_value = clip_value
    self.device = device
    self.name = name

  def train_epoch(self, central_model):
    local_model = copy.deepcopy(central_model)
    optimizer = self.optimizer
    if callable(optimizer):
      optimizer = optimizer(local_model.parameters())  # stateless optimizer
    outputs, meters = train_epoch(
      model=local_model,
      optimizer=optimizer,
      forward_pass_fn=forward_pass,
      dataloader=self.train_loader,
      steps_per_epoch=self.steps_per_epoch,
      clip_value=self.clip_value,
      device=self.device)
    update = [updated_param.data - initial_param.data  # difference between the local model and the central model
              for updated_param, initial_param in zip(local_model.parameters(), central_model.parameters())]
    return update, outputs, meters

  def eval_epoch(self, central_model):  # evaluates the central model
    return eval_epoch(
      model=central_model,
      forward_pass_fn=forward_pass,
      dataloader=self.val_loader,
      device=self.device)


class MRMTLClient:
  # computes an update of the local model on the local data;
  #  local model is penalized for diverging from the central model
  def __init__(self, local_model, optimizer, kappa, train_loader, val_loader,
               steps_per_epoch=None, clip_value=None, device=None, name=None):
    if not callable(optimizer) and not isinstance(optimizer, torch.optim.Optimizer):
      raise ValueError(f'Unknown optimizer: {optimizer}')
    self.local_model = local_model
    self.optimizer = optimizer
    self.kappa = kappa
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.steps_per_epoch = steps_per_epoch
    self.clip_value = clip_value
    self.device = device
    self.name = name

  def train_epoch(self, central_model):
    initial_local_model = copy.deepcopy(self.local_model)
    optimizer = self.optimizer
    if callable(optimizer):
      optimizer = optimizer(self.local_model.parameters())  # stateless optimizer
    outputs, meters = train_epoch(
      model=self.local_model,
      optimizer=optimizer,
      forward_pass_fn=self.forward_pass(central_model),  # custom forward pass
      dataloader=self.train_loader,
      steps_per_epoch=self.steps_per_epoch,
      clip_value=self.clip_value,
      device=self.device)
    update = [updated_param.data - initial_param.data  # difference between the updated and the initial local model
              for updated_param, initial_param in zip(self.local_model.parameters(), initial_local_model.parameters())]
    return update, outputs, meters

  # noinspection PyUnusedLocal
  def eval_epoch(self, central_model):  # evaluates the local model
    del central_model
    return eval_epoch(
      model=self.local_model,
      forward_pass_fn=forward_pass,
      dataloader=self.val_loader,
      device=self.device)

  def forward_pass(self, central_model):
    def forward_pass_during_training(model, batch):
      batch_loss, (logits, targets) = forward_pass(model, batch)
      diff_from_central = [local_param.data - central_param.data  # difference between the local and the central model
                           for local_param, central_param in zip(model.parameters(), central_model.parameters())]
      local_divergence_penalty = global_norm(diff_from_central) ** 2
      total_loss = batch_loss + self.kappa / 2 * local_divergence_penalty
      return total_loss, (logits, targets)
    return forward_pass_during_training


def train_epoch(model, optimizer, forward_pass_fn, dataloader,
                steps_per_epoch=None, clip_value=None, device=None):
  """
  Exemplary `forward_pass` function:

    def forward_pass(model, batch):
      inputs, targets = batch
      logits = model(inputs)
      batch_loss = loss_fn(logits, targets)
      return batch_loss, (logits, targets)
  """
  model.train()
  if steps_per_epoch is not None:  # limit the number of steps
    dataloader = itertools.islice(dataloader, steps_per_epoch)
  if device is not None:  # move every batch to the device
    dataloader = map(functools.partial(map_to_device, device=device), dataloader)
  step_meter = utils.AverageMeter('train_step')
  data_meter = utils.AverageMeter('train_data')
  loss_meter = utils.LossMeter('train_loss')
  train_logits = []
  train_targets = []
  step_start = time()
  for batch in dataloader:
    data_end = time()
    batch_loss, (logits, targets) = forward_pass_fn(model, batch)
    optimizer.zero_grad()
    batch_loss.backward()
    if clip_value is not None:  # required in FL because the gradients sometimes explode early on
      nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
    optimizer.step()
    step_end = time()
    loss_meter.update(batch_loss.item())
    step_meter.update(step_end - step_start)
    data_meter.update(data_end - step_start)
    train_logits.append(logits.detach().cpu())
    train_targets.append(targets.cpu())
    step_start = time()
  train_logits = torch.cat(train_logits)
  train_targets = torch.cat(train_targets)
  return (train_logits, train_targets), (step_meter, data_meter, loss_meter)


@torch.no_grad()
def eval_epoch(model, forward_pass_fn, dataloader, device=None):
  """
  Exemplary `forward_pass` function:

    def forward_pass(model, batch):
      inputs, targets = batch
      logits = model(inputs)
      return (logits, targets)
  """
  model.eval()
  if device is not None:  # move every batch to the device
    dataloader = map(functools.partial(map_to_device, device=device), dataloader)
  eval_logits = []
  eval_targets = []
  for batch in dataloader:
    logits, targets = forward_pass_fn(model, batch)
    eval_logits.append(logits.cpu())
    eval_targets.append(targets.cpu())
  eval_logits = torch.cat(eval_logits)
  eval_targets = torch.cat(eval_targets)
  return eval_logits, eval_targets


def forward_pass(model, batch):
  x, _, target_mask, targets = batch
  logits = model(x, target_mask).masked_fill(target_mask == 0, -1e9)  # mask nonexistent diagnoses
  if model.training:
    batch_loss = F.binary_cross_entropy_with_logits(logits, targets)
    return batch_loss, (logits, targets)
  else:
    return logits, targets


def update_lr_with_cosine_schedule_(
    optimizer,
    step: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 0.,
    warmup_steps: int = 0
) -> None:
  if step < warmup_steps:
    lr = max_lr * step / warmup_steps
  elif step > max_steps:
    lr = min_lr
  else:  # cosine learning rate schedule
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps - 1)
    coefficient = 0.5 * (1. + math.cos(math.pi * decay_ratio))
    lr = min_lr + coefficient * (max_lr - min_lr)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def weight_clients(training_sets, device=None):
  """Weights clients by the size of training set."""
  train_sizes = torch.tensor([len(training_set) for training_set in training_sets], device=device)
  client_weights = train_sizes / train_sizes.sum()
  return client_weights


def remove_bias_correction_(optim: torch.optim.Adam, eps: float = 1e-3) -> None:
  """Remove bias correction from Adam optimizer by setting high enough step."""
  for param_group in optim.param_groups:
    beta1, beta2 = param_group['betas']
    larger_beta = max(beta1, beta2)
    min_step = math.ceil(math.log(eps) / math.log(larger_beta))
    for param in param_group['params']:
      state = optim.state[param]
      if len(state) == 0:
        optim.state[param] = {
          'step': min_step,
          'exp_avg': torch.zeros_like(param, memory_format=torch.preserve_format),
          'exp_avg_sq': torch.zeros_like(param, memory_format=torch.preserve_format)
        }
      elif state['step'] < min_step:
        state['step'] = min_step


@torch.no_grad()
def gradient_metrics(list_of_gradients, weights):
  global_grad_norm = torch.tensor(0., device=weights.device)  # gradient magnitude
  global_grad_var = torch.tensor(0., device=weights.device)  # gradient variance
  for gradients in stack_gradients(list_of_gradients):
    w_mean, w_var = weighted_moments(gradients, weights)
    global_grad_norm += torch.norm(w_mean, p=2) ** 2
    global_grad_var += w_var.sum()
  global_grad_norm = global_grad_norm.sqrt()
  return global_grad_norm, global_grad_var


def weighted_moments(x, weights, unbiased=True):
  assert x.size(0) == weights.size(0)
  assert weights.ndim == 1
  weights = weights / weights.sum()
  weights = weights.reshape(-1, *(1,) * (x.ndim - 1))
  w_mean = torch.sum(weights * x, dim=0)
  w_var = torch.sum(weights * (x - w_mean.unsqueeze(0)) ** 2, dim=0)
  if unbiased:
    bias = 1 - (weights ** 2).sum() / weights.sum() ** 2
    w_var = w_var / bias
  return w_mean, w_var


def stack_gradients(list_of_gradients):
  return [torch.stack(gradients) for gradients in zip(*list_of_gradients)]


@torch.no_grad()
def global_norm(tensors):
  return torch.stack(
    [torch.norm(tensor, p=2) ** 2 for tensor in tensors]
  ).sum().sqrt()


def split_data(db_path, data, val=None):
  db_name, *_ = path.basename(db_path).split('.')
  (x, input_mask, target_mask), targets, record_names = data

  # load train / test split
  data_split_file = path.join('evaluation', f'split_{db_name}.csv')
  data_split_df = pd.read_csv(data_split_file, index_col='record')

  # match records in the split file with record files
  record_index = {record_name: index for index, record_name in enumerate(record_names)}
  data_split_df['record_index'] = data_split_df.index.map(record_index.get)

  missing_records = data_split_df[data_split_df['record_index'].isna()].index.to_numpy()

  if len(missing_records) > 0:
    logging.warning(f'Missing {len(missing_records)} '
                    f'record{"s" if len(missing_records) > 1 else ""}: '
                    f'{", ".join(missing_records)}')

  # create training and test sets
  train_idx = data_split_df['record_index'][data_split_df['train']].dropna().to_numpy(dtype=np.int32)
  test_idx = data_split_df['record_index'][data_split_df['test']].dropna().to_numpy(dtype=np.int32)

  # sample train records to form a validation set
  if val is not None:
    adjusted_val_size = val * (1. + len(test_idx) / len(train_idx))
    train_idx, _, val_idx, _ = iterative_train_test_split(
      train_idx[:, np.newaxis], targets[train_idx], adjusted_val_size)
    train_idx = train_idx.squeeze(axis=1)
    val_idx = val_idx.squeeze(axis=1)
  else:
    val_idx = None

  return DataSplit(
    db_name=db_name,
    dataset=TensorDataset(x, input_mask, target_mask, targets),
    record_names=np.array(record_names),
    training_idx=train_idx,
    validation_idx=val_idx,
    test_idx=test_idx)


@dataclass
class DataSplit:
  db_name: str
  dataset: torch.utils.data.Dataset
  record_names: np.array
  training_idx: Optional[np.array] = None
  validation_idx: Optional[np.array] = None
  test_idx: Optional[np.array] = None


class RecordPreprocessor:
  def __init__(self, duration=60, fs=128):
    self.select_leads = transforms.SelectLeads(physionet2021.LEADS)
    self.replace_NaN = transforms.ReplaceNaN()
    self.resample = transforms.Resample(fs)
    self.normalize = transforms.Normalize()
    self.crop = transforms.CenterCrop(fs * duration)
    self.pad = transforms.MaskedPad(fs * duration)
    self.to_tensor = transforms.ToTensor(channels_first=True)

  def __call__(self, record: Record) -> Tuple[Tensor, Tensor]:
    record = self.select_leads(record)
    record = self.replace_NaN(record)
    record = self.resample(record)
    record = self.normalize(record)
    record = self.crop(record)
    record, mask = self.pad(record)
    record = self.to_tensor(record)
    return record, mask


def map_to_device(x, device: torch.device):
  return apply_to_tensor(lambda tensor: tensor.to(device), x)


def series_to_tensor(series):
  return torch.from_numpy(series.to_numpy())


def apply_to_tensor(fn, x):
  if isinstance(x, Tensor):
    return fn(x)
  elif isinstance(x, (tuple, list)):
    container = type(x)
    return container(apply_to_tensor(fn, e) for e in x)
  elif isinstance(x, dict):
    return {key: apply_to_tensor(fn, value) for key, value in x.items()}
  elif x is None:
    return None
  else:
    raise ValueError(f'unknown data type: {type(x)}')


if __name__ == '__main__':
  logging.config.fileConfig('logging.ini')
  arguments = parser.parse_args()
  main(arguments)
  # indicate that the job has finished successfully
  open(path.join(arguments.job_dir, '.success'), 'w').close()
