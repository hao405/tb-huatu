if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/run" ]; then
    mkdir ./logs/run
fi

if [ ! -d "./logs/run/test" ]; then
    mkdir ./logs/run/test
fi

model_name=TimeBridge
seq_len=720
GPU=0
root=./dataset

alpha=0.34049992849329375
data_name=ETTm2
for pred_len in 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --pd_layers 1 \
    --enc_in 7 \
    --ca_layers 0 \
    --pd_layers 1 \
    --ia_layers 3 \
    --des 'Exp' \
    --n_heads 4 \
    --d_model 64  \
    --d_ff 128 \
    --lradj 'TST' \
    --period 48 \
    --train_epochs 100 \
    --learning_rate 0.00035442840646088774 \
    --pct_start 0.2 \
    --patience 10 \
    --batch_size 32 \
    --alpha $alpha \
    --itr 1 | tee logs/run/test/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

#  - learning_rate: 0.0003238187768686929
#  - batch_size: 64
#  - n_heads: 4
#  - alpha: 0.34588140818892216
#  - d_ff_multiplier: 4

#  - learning_rate: 0.00035442840646088774
#  - batch_size: 32
#  - n_heads: 4
#  - alpha: 0.34049992849329375