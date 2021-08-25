#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

if [ "$#" -ne 4 ]; then
    echo "Help: bash run_unsup_example.sh {bpe_dropout_prob} {learning_rate} {max_seq_length} {batch_size}"
    exit 2
fi

num_gpu=2
master_port=10001

bpe_dropout_prob=$1
learning_rate=$2
max_seq_length=$3
batch_size=$4
gradient_accumulation_step=$((batch_size / (64 * num_gpu)))

echo "bpe_dropout_prob: ${bpe_dropout_prob}"
echo "learning_rate: ${learning_rate}"
echo "max_seq_length: ${max_seq_length}"
echo "batch_size: ${batch_size}"
echo "gradient_accumulation_step: ${gradient_accumulation_step}"

# Tokenize corpus
python -m run_tokenize \
    --filepath .data/wiki1m_for_simcse.txt \
    --bpe-dropout-prob ${bpe_dropout_prob} \


# Allow multiple threads
export OMP_NUM_THREADS=8

python -m run_train_simcse \
    --model_name_or_path roberta-base \
    --train_file .data/wiki1m_for_simcse.txt_bpedropout_${bpe_dropout_prob}_roberta-base.csv \
    --output_dir result/my-unsup-simcse-roberta-base-${bpe_dropout_prob}-${learning_rate}-${max_seq_length} \
    --overwrite_output_dir \
    --dataloader_drop_last \
    --num_train_epochs 1 \
    --learning_rate ${learning_rate} \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_step ${gradient_accumulation_step} \
    --max_seq_length ${max_seq_length} \
    --evaluation_strategy steps \
    --eval_steps 125 \
    --metric_for_best_model stsb_spearman \
    --save_strategy steps \
    --save_steps 125 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --pooler_type cls_before_pooler \
    --mlp_only_train \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16

python -m evaluation \
    --model_name_or_path result/my-unsup-simcse-roberta-base-${bpe_dropout_prob}-${learning_rate}-${max_seq_length} \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test
