dup_rate_list=("0.08", "0.16" "0.24" "0.32" "0.40")
seed_list=(3)
learning_rate_list=("4e-5")
layer_idx_list=(12)

train_file="/path/to/nli_for_simcse.csv"
eval_file="/path/to/eval/directory"

for seed in ${seed_list[@]}; do
for learning_rate in ${learning_rate_list[@]}; do
for layer_idx in ${layer_idx_list[@]}; do
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python \
    src/scripts/run_train.py \
    --model_name_or_path roberta-base \
    --loss_type simcse-avg \
	  --data_type simcse-nli \
    --train_file ${train_file} \
    --eval_file ${eval_file} \
    --dataloader_drop_last \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --mlp_only_train \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --num_train_epochs 3 \
    --save_total_limit 1 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${learning_rate} \
    --hidden_dropout_prob 0.1 \
    --seed ${seed} \
    --layer_idx ${layer_idx} \
    --dense_rwmd

done
done
done
