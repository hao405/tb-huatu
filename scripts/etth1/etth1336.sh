if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/test" ]; then
    mkdir ./logs/test
fi

if [ ! -d "./logs/test/optune" ]; then
    mkdir ./logs/test/optune
fi

model_name=TimeBridge
seq_len=720
GPU=0
root=./dataset

alpha=0.35
data_name=ETTh1
for pred_len in 336
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
    --enc_in 7 \
    --ca_layers 0 \
    --pd_layers 1 \
    --ia_layers 3 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 512 \
    --alpha $alpha \
    --learning_rate 0.0001772803774288082 \
    --train_epochs 100 \
    --patience 10 \
    --attn_dropout 0.15 \
    --n_heads 64 \
    --num_p 4 \
    --batch_size 8 \
    --itr 1 | tee logs/test/optune/$data_name'_'$alpha'_'$model_name'_'$pred_len'_'$attn_dropout.logs
done