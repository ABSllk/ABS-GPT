# ABS-GPT

一个学习项目：用 NumPy / CuPy 手写自动微分和 Transformer，探索语言模型的训练与文本生成。核心实现不依赖 PyTorch。

包含预训练、SFT、LoRA，以及 GQA、MoE、MTP、Muon 等功能，GPU 算子使用 CUDA / Triton。

```text
core/        自动微分、网络层、优化器与 GPU 算子
models/      语言模型
engine/      训练、学习率调度、检查点与生成
dataset/     数据加载与 token 缓存
tokenizers/  字符与 BPE 分词器
scripts/     训练、微调、生成与绘图入口
configs/     训练配置
```

当前训练入口需要 CUDA 环境，统一安装依赖：

```bash
pip install -r requirements.txt
```

依赖默认使用 CUDA 12 版 CuPy；CUDA 13 环境将其中的 `cupy-cuda12x` 换成 `cupy-cuda13x`。

预训练使用含 `text` 列的 Parquet 文件，SFT 使用含 `prompt`、`completion` 列的 Parquet 文件。调整 `configs/` 中的配置后，在项目根目录运行：

```bash
python -m scripts.train_transformer --run-dir results/my-run
python -m scripts.generate_transformer --run-dir results/my-run "春江花月夜"
```

微调和其他参数见 `python -m scripts.train_sft --help` 等入口帮助。
