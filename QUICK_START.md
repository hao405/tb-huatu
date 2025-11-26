# DDP 多GPU训练 - 快速开始

## 🎯 问题解决

原始错误：
```
RuntimeError: Trying to access a forward AD level with an invalid index
TypeError: cannot unpack non-iterable NoneType object
```

**根本原因**：`DataParallel` 与 `functorch` 的 `jacfwd`/`vmap` 不兼容

**解决方案**：迁移到 `DistributedDataParallel (DDP)`

## 🚀 快速开始

### 1. 直接运行 traffic.sh（最简单）

```bash
cd /Users/zhuhao/experiment/first_learning/TB-96/TimeBridge-v3-xianyan
bash scripts/traffic.sh
```

### 2. 或使用通用脚本

```bash
bash scripts/run_ddp_tune.sh
```

### 3. 测试 DDP 设置（推荐先运行）

```bash
bash scripts/test_ddp.sh
```

## 📋 已修改的文件

1. ✅ `experiments/exp_basic.py` - 添加 DDP device 管理
2. ✅ `experiments/exp_long_term_forecasting.py` - 使用 DDP 替代 DataParallel  
3. ✅ `data_provider/data_factory.py` - 添加 DistributedSampler
4. ✅ `tune_big.py` - 添加 DDP 初始化和清理
5. ✅ `scripts/traffic.sh` - 使用 torchrun 启动
6. ✅ `model/TimeBridge.py` - 保持原样（DDP 下无需修改）

## 🔧 关键变化

### 之前（DataParallel）
```bash
python -u tune_big.py --use_multi_gpu --devices 0,1,2,3
```

### 现在（DDP）
```bash
torchrun --nproc_per_node=4 --master_port=29500 tune_big.py --use_multi_gpu --devices 0,1,2,3
```

## ⚙️ 重要参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--nproc_per_node` | 每个节点的GPU数量 | `8` |
| `--master_port` | 主进程通信端口 | `29500` |
| `--use_multi_gpu` | 启用多GPU | 已默认开启 |
| `--devices` | GPU列表 | `0,1,2,3,4,5,6,7` |
| `--batch_size` | 单GPU批次大小 | `4`（8GPU时有效批次=32） |

## 🎮 AMD ROCm 特定设置

```bash
# 指定使用的 GPU
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 禁用 MIOpen 缓存
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_SYSTEM_DB_PATH=""
```

这些已在 `traffic.sh` 中自动设置。

## 📊 批次大小计算

**有效批次大小 = batch_size × GPU数量**

示例：
- `--batch_size 4` × 8个GPU = **有效批次 32**
- `--batch_size 8` × 4个GPU = **有效批次 32**

根据显存调整 `batch_size`。

## ✅ 验证 DDP 是否工作

运行后应该看到：
```
Use GPU (DDP): cuda:0, Rank: 0/8
Use GPU (DDP): cuda:1, Rank: 1/8
...
Use GPU (DDP): cuda:7, Rank: 7/8
```

检查所有 GPU 是否都在工作：
```bash
watch -n 1 rocm-smi
```

## 🐛 常见问题

### 端口占用
```bash
# 更改端口
torchrun --master_port=29501 ...
```

### 内存不足
```python
# 减小 batch_size
--batch_size 2  # 而不是 4
```

### NCCL 错误
```bash
# 使用 gloo backend（较慢但更稳定）
export TORCH_DISTRIBUTED_BACKEND=gloo
```

## 📚 详细文档

查看 `DDP_GUIDE.md` 获取完整文档。

## 🎉 完成！

现在可以使用多GPU训练了，不会再出现 functorch 错误！
