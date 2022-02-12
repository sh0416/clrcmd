Contrastive Learning: Relaxed Contextualized word Mover Distance (CLRCMD)
==================

This repository reproduces the experimental result reported in the paper.

## 1. Prepare Environment
We assume that the user uses anaconda environment.
```
conda create -n clrcmd python=3.8
conda activate clrcmd
pip install -r requirements.txt
python setup.py develop
```

## 2. Prepare dataset

### 2-1. Semantic Textual Similarity benchmark (STS12, STS13, STS14, STS15, STS16, STSB, SICKR)
We download the benchmark dataset using the script provided by SimCSE repository.  
```
bash examples/download_sts.sh
```
* `tokenizer.sed`: Tokenizer script used in `download_sts.bash`

### 2-2. Interpretable Semantic Textual Similarity benchmark (iSTS)
We create a script for downloading iSTS benchmarks.
```
bash examples/download_ists.sh
```

#### 2-2-1. Correct wrong input format
* `STSint.testinput.answers-students.sent1.chunk.txt`
 * 252th example: from `a closed path` to `a closed path.`
 * 287th example: from `has no gaps` to `[ has no gaps ]`
 * 315th example: from `is in a closed path,` to `[ is in a closed path, ]`
 * 315th example: from `is in a closed path.` to `[ is in a closed path. ]`
* `STSint.testinput.answers-students.sent1.txt`
 * 287th example: `battery  terminal` to `battery terminal`
 * 308th example: `switch z,  that` to `switch z, that`
* `STSint.testinput.answers-students.sent2.chunk.txt`
 * 287th example: `are not separated by the gap` to `[ are not separated by the gap ]`
 * 315th example: `are` to `[ are ]`
 * 315th example: `in closed paths` to `[ in closed path ]`

### 2-3. NLI dataset tailored for self-supervised learning (SimCSE-NLI)
We download the training dataset using the script provided by SimCSE repository.
```
bash examples/download_nli.bash
```

## 3. Conduct experiments

### 3-1. Evaluate semantic textual similarity benchmark without any training
```
# Help message
python examples/run_evaluate.py -h

# One example
python examples/run_evaluate.py --data-dir data --dataset sts12 --model bert-cls
```

### 3-2. Train model using self-supervised learning (e.g. SimCSE, CLRCMD)
```
python examples/run_train.py --data-dir data --model bert-cls
```

### 3-2. Evaluate benchmark performance on the trained checkpoint
```
python examples/run_evaluate.py --data-dir data --dataset sts12 --model bert-cls --checkpoint ckpt/bert-cls/checkpoint-best
```

## 4. Report results

### 4-1. Semantic textual similarity benchmark

```
optuna create-study --storage sqlite:///experiments.db --study-name train --direction maximize
optuna studies --storage sqlite:///experiments.db
optuna trials --storage sqlite:///experiments.db --study-name train --flatten
CUDA_VISIBLE_DEVICES=2 optuna study optimize examples/run_tune.py objective --n-trials 20 --storage sqlite:///experiments.db --study-name train
CUDA_VISIBLE_DEVICES=3 optuna study optimize examples/run_tune.py objective --n-trials 20 --storage sqlite:///experiments.db --study-name train
```