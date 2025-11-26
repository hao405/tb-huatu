if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/test" ]; then
    mkdir ./logs/test
fi

if [ ! -d "./logs/test/new" ]; then
    mkdir ./logs/test/new
fi

model_name=TimeBridge
seq_len=96
GPU=0,1,2,3,4,5,6,7
root=./data

alpha=0.35
data_name=traffic
for pred_len in  720 96 192 336;
do
  MIOPEN_DISABLE_CACHE=1 \
  MIOPEN_SYSTEM_DB_PATH="" \
  HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  python -u tune_big.py \
    --is_training 1 \
    --root_path $root/traffic/ \
    --data_path traffic.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --num_p 8 \
    --n_heads 64 \
    --stable_len 2 \
    --d_ff 512 \
    --d_model 512 \
    --ca_layers 3 \
    --pd_layers 1 \
    --ia_layers 1 \
    --batch_size 4 \
    --attn_dropout 0.15 \
    --patience 10 \
    --train_epochs 100 \
    --devices 0,1,2,3,4,5,6,7 \
    --alpha $alpha \
    --learning_rate 0.0005 \
    --itr 1 | tee logs/test/new/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

#alpha=0.35
#data_name=traffic
#GPU=0,1,2,3
#for pred_len in 336 720 192 96; do
#  seq_len=$((2 * pred_len))
#  CUDA_VISIBLE_DEVICES=$GPU \
#  python -u tune.py \
#    --is_training 1 \
#    --root_path $root/traffic/ \
#    --data_path traffic.csv \
#    --model_id $data_name'_'$seq_len'_'$pred_len \
#    --model $model_name \
#    --data custom \
#    --features M \
#    --seq_len $seq_len \
#    --label_len 48 \
#    --pred_len $pred_len \
#    --enc_in 862 \
#    --des 'Exp' \
#    --num_p 8 \
#    --n_heads 64 \
#    --stable_len 2 \
#    --d_ff 512 \
#    --d_model 512 \
#    --ca_layers 3 \
#    --pd_layers 1 \
#    --ia_layers 1 \
#    --batch_size 4 \
#    --attn_dropout 0.15 \
#    --patience 5 \
#    --train_epochs 100 \
#    --devices 0,1,2,3 \
#    --use_multi_gpu \
#    --alpha $alpha \
#    --learning_rate 0.0005 \
#    --itr 1 | tee logs/LongForecasting/TimeBridge3/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
#done