Sentence Benchmark
==================

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --np$
oc_per_node 4 --master_port 10001 run_train_simcse.py --model_name_or_path roberta-base --train_file .data/wiki1m_for_simcs$
.txt_bpedropout_0.0_roberta-base.csv --dataloader_drop_last --evaluation_strategy steps --eval_steps 125 --metric_for_best_$
odel stsb_spearman --load_best_model_at_end --mlp_only_train --temp 0.05 --do_train --do_eval --fp16 --num_train_epochs 1 -$
save_total_limit 1 --loss_token --coeff_loss_token 1 --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --gr$
dient_accumulation_steps 2 --learning_rate 1e-5 --hidden_dropout_prob 0.1
```

STS12, STS13, STS14, STS15, STS16, STS-B, SICK-R를 평가할 수 있는 벤치마크를 구성한다.

많은 부분 SentEval 레포지토리를 따라하지만 SentEval과 다르게 데이터셋을 쉽게 추가할 수 있게 만든다.

```bash
# SentEval benchmark data loading script
bash get_transfer_data.bash
# SimCSE training data loading script
wget -O .data/wiki1m_for_simcse.txt https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt
conda create -n sentence-benchmark
conda activate sentence-benchmark
conda install numpy scipy scikit-learn matplotlib
pip install -r requirements.txt
python -m run_evaluate
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

