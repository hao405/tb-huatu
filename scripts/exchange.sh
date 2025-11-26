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
GPU=5
root=./data
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export HIP_VISIBLE_DEVICES=$GPU
alpha=0.35
data_name=exchange_rate
for pred_len in 96 192 336 720
do
  HIP_VISIBLE_DEVICES=$GPU \
  python -u tune2.py \
    --is_training 1 \
    --root_path $root/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 8 \
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
    --itr 1 | tee logs/test/new/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

#alpha=0.2
#data_name=exchange_rate
#for pred_len in 48 96 144 192
#do
#  seq_len=$((2 * pred_len))
#  CUDA_VISIBLE_DEVICES=$GPU \
#  python -u tune.py \
#    --is_training 1 \
#    --root_path $root/exchange_rate/ \
#    --data_path exchange_rate.csv \
#    --model_id $data_name'_'$seq_len'_'$pred_len \
#    --model $model_name \
#    --data custom \
#    --features M \
#    --seq_len $seq_len \
#    --label_len 48 \
#    --pred_len $pred_len \
#    --enc_in 8 \
#    --des 'Exp' \
#    --n_heads 32 \
#    --d_ff 512 \
#    --d_model 512 \
#    --ca_layers 2 \
#    --pd_layers 1 \
#    --ia_layers 1 \
#    --attn_dropout 0.1 \
#    --num_p 4 \
#    --stable_len 4 \
#    --alpha $alpha \
#    --batch_size 16 \
#    --learning_rate 0.0005 \
#    --itr 1 | tee logs/LongForecasting/TimeBridge3/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
#done

