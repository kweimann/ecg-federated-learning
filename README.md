# Federated learning improves performance of ECG classifiers

We explore federated learning of deep residual networks to diagnose cardiac abnormalities based on electrocardiogram data. We compare three federated learning methods with central training on the public data from the PhysioNet/Computing in Cardiology Challenge 2021. Our findings demonstrate the benefits of federated learning methods, in particular the FedOpt algorithm. A global model trained in federation and finetuned to the local dataset outperforms non-collaborative methods on the local test sets of participating clients. This shows that models trained in federation learn general features that can be tailored to specific tasks. Furthermore, federated learning almost matches the performance of central training with data sharing on out-of-distribution data from clients that did not participate in the training. This result emphasizes the ability of federated learning to train models that generalize well across diverse patient data, without the need to share data among institutions, thus addressing data privacy concerns.

### Usage

Preliminaries:
* Install dependencies: `pip install -r requirements.txt`
* Download data from https://physionet.org/content/challenge-2021/1.0.3/#files and put it in `data/`

#### Centrally train and evaluate a ResNet on data from 4 out of 5 clients:

```bash
python train.py \
  --job-dir=Central \
  --train-db=data/WFDB_CPSC2018 data/WFDB_CPSC2018_2 data/WFDB_ChapmanShaoxing data/WFDB_Ningbo data/WFDB_PTB data/WFDB_PTBXL \
  --test-db=data/WFDB_CPSC2018 data/WFDB_CPSC2018_2 data/WFDB_ChapmanShaoxing data/WFDB_Ningbo data/WFDB_PTB data/WFDB_PTBXL \
  --cache-dir=.cache \
  --optimization=central \
  --epochs=100
```

#### Train and evaluate the global model using FedOpt on data from 4 out of 5 clients:

```bash
python train.py \
  --job-dir=FedOpt \
  --train-db=data/WFDB_CPSC2018 data/WFDB_CPSC2018_2 data/WFDB_ChapmanShaoxing data/WFDB_Ningbo data/WFDB_PTB data/WFDB_PTBXL \
  --test-db=data/WFDB_CPSC2018 data/WFDB_CPSC2018_2 data/WFDB_ChapmanShaoxing data/WFDB_Ningbo data/WFDB_PTB data/WFDB_PTBXL \
  --cache-dir=.cache \
  --optimization=FedOpt \
  --fl-adaptive-opt \
  --server-lr=0.001 \
  --server-beta1=0.99 \
  --server-beta2=0.999 \
  --client-lr=0.1 \
  --attention-pooling \
  --uniform-client-weights \
  --epochs=300
```

#### Evaluate the global model on out-of-distribution data from the client that did not participate (`data/WFDB_Ga`):

```bash
python train.py \
  --job-dir=FedOpt-eval \
  --test-db=data/WFDB_Ga \
  --cache-dir=.cache \
  --checkpoint=FedOpt/checkpoint.pth \
  --no-split-test \
  --attention-pooling
```

#### Finetune the global model on data from `data/WFDB_Ga`:

```bash
python train.py \
  --job-dir=FedOpt-finetuned \
  --test-db=data/WFDB_Ga \
  --cache-dir=.cache \
  --checkpoint=FedOpt/checkpoint.pth \
  --optimization=central \
  --cosine-lr-schedule \
  --epochs=100
```

#### Train a model on data from `data/WFDB_Ga` without collaboration:

```bash
python train.py \
  --job-dir=Central-single \
  --train-db=data/WFDB_Ga \
  --test-db=data/WFDB_Ga \
  --cache-dir=.cache \
  --optimization=central \
  --epochs=100
```