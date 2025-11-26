if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/optune" ]; then
    mkdir ./logs/optune
fi

if [ ! -d "./logs/optune/tune2" ]; then
    mkdir ./logs/optune/tune2
fi

model_name=TimeBridge
seq_len=720
GPU=1
root=./dataset

alpha=0.35
data_name=ETTh1
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u tune2.py \
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
    --d_ff 128 \
    --batch_size 64 \
    --alpha $alpha \
    --learning_rate 0.0002 \
    --train_epochs 100 \
    --patience 10 \
    --itr 1 | tee logs/optune/tune2/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done