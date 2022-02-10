Contrastive Learning: Relaxed Contextualized word Mover Distance (CLRCMD)
==================

This repository reproduces the experimental result reported in the paper.

## 1. Prepare Environment
We assume that the user uses anaconda environment.
```
conda create -n clrcmd python=3.8
conda activate clrcmd
pip install -r requirements.txt
```

## 2. Prepare dataset

### 2-1. Semantic Textual Similarity benchmark (STS12, STS13, STS14, STS15, STS16, STSB, SICKR)
We download the benchmark dataset using the script provided by SimCSE repository.  
```
bash examples/download_sts.sh
```

### 2-2. Interpretable Semantic Textual Similarity benchmark (iSTS)
We create a script for downloading iSTS benchmarks.
```
bash examples/download_ists.sh
```


### 2-3. NLI dataset tailored for self-supervised learning (SimCSE-NLI)
We download the training dataset using the script provided by SimCSE repository.
```
bash examples/download_nli.bash
```

## 3. Evaluate sentence similarity benchmark

### 3-1. Evaluate benchmark performance on pretrained checkpoint
```
torchrun examples/run_evaluate.py --data-dir data --dataset sts12 --model bert-cls
torchrun examples/run_evaluate.py --data-dir data --dataset sts12 --model bert-avg
torchrun examples/run_evaluate.py --data-dir data --dataset sts12 --model roberta-cls
torchrun examples/run_evaluate.py --data-dir data --dataset sts12 --model roberta-avg
```

### 3-2. Evaluate benchmark performance on the trained checkpoint
```
torchrun examples/run_evaluate.py --data-dir data --dataset sts12 --model bert-cls --checkpoint ckpt/simcse.pt
torchrun examples/run_evaluate.py --data-dir data --dataset sts12 --model bert-avg --checkpoint ckpt/simcse.pt
torchrun examples/run_evaluate.py --data-dir data --dataset sts12 --model bert-rcmd --checkpoint ckpt/clrcmd.pt
```

### 3-3. Evaluate benchmark performance on the checkpoint trained on CLRCMD


# Legacy README
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src python -m torch.distributed.launch --nproc_per_node 4 --master_port 10001 src/scripts/run_train.py --model_name_or_path roberta-base --train_file data/wiki1m_for_simcse.txt_bpedropout_0.0_roberta-base.csv --dataloader_drop_last --evaluation_strategy steps --eval_steps 125 --metric_for_best_model stsb_spearman --load_best_model_at_end --mlp_only_train --temp 0.05 --do_train --do_eval --fp16 --num_train_epochs 1 --save_total_limit 1 --loss_token --coeff_loss_token 1 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --learning_rate 1e-5 --hidden_dropout_prob 0.1
```

STS12, STS13, STS14, STS15, STS16, STS-B, SICK-R를 평가할 수 있는 벤치마크를 구성한다.

많은 부분 SentEval 레포지토리를 따라하지만 SentEval과 다르게 데이터셋을 쉽게 추가할 수 있게 만든다.

```bash
cd src/scripts
bash get_transfer_data.bash
bash download_nli.sh
cd ../..
conda create -n clrcmd
conda activate clrcmd
conda install numpy scipy scikit-learn matplotlib
pip install -r requirements.txt
bash train_sts.sh
```

## Files

### Dataset
* `get_transfer_data.bash`: Download senteval data from web and preprocess it (e.g. mosestokenizer)
* `tokenizer.sed`: Tokenizer script used in `get_transfer_data.bash`

### Model
* `simcse/*`: Code related to the simcse paper
* `tokenizer.py`: Tokenizer that the bpe dropout is implemented in this script
* `run_train.py`: Finetune model using dataset

### Evaluate
* `run_evaluate.py` : Evaluate sentence representation quality using given benchmark data


### Change Log
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
