if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/optune" ]; then
    mkdir ./logs/optune
fi

if [ ! -d "./logs/optune/tune3" ]; then
    mkdir ./logs/optune/tune3
fi

model_name=TimeBridge
seq_len=720
GPU=0
root=./dataset

alpha=0.1
data_name=weather
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u tune3.py \
    --is_training 1 \
    --root_path $root/weather/ \
    --data_path weather.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 21 \
    --ca_layers 1 \
    --pd_layers 1 \
    --ia_layers 1 \
    --des 'Exp' \
    --period 48 \
    --num_p 12 \
    --d_model 128 \
    --d_ff 128 \
    --alpha $alpha \
    --itr 1 | tee logs/optune/tune3/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done