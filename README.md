# Qwen QLoRA 微调模板

基于 Qwen3.5-0.8B 模型的低秩微调训练与推理项目，支持 LoRA/QLoRA 训练方式。

---

## 📁 项目结构

```
├── train.py              # 训练脚本
├── inference.py          # 命令行推理脚本
├── app.py                # Gradio Web 界面
├── run_app.py            # Web 启动脚本
├── dataset.jsonl         # 训练数据集
├── requirements.txt      # 依赖列表
├── qwen-lora/            # 训练输出目录（自动创建）
└── README.md
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据集

编辑 `dataset.jsonl`，每行格式：

```json
{"messages": [{"role": "system", "content": "你是一个Python专家"}, {"role": "user", "content": "如何定义函数？"}, {"role": "assistant", "content": "使用def关键字..."}]}
```

### 3. 开始训练

```bash
python train.py
```

训练完成后模型保存在 `qwen-lora/` 目录。

---

## 🎯 使用方式

### 方式一：命令行对话

```bash
python inference.py
```

**快捷命令：**
| 命令 | 说明 |
|------|------|
| `help` | 显示帮助 |
| `clear` | 清空对话 |
| `history` | 查看历史 |
| `save` | 保存对话 |
| `load` | 加载对话 |
| `config` | 调整参数 |
| `system` | 修改系统提示 |
| `exit` | 退出 |

### 方式二：Web 界面

```bash
python app.py
# 或双击
run_app.py
```

启动后访问 **http://127.0.0.1:7860**

---

## ⚙️ 配置说明

### 训练参数（train.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BASE_MODEL` | `Qwen/Qwen3.5-0.8B` | 基础模型 |
| `LORA_PATH` | `./qwen-lora` | LoRA 保存路径 |
| `DATASET_PATH` | `./dataset.jsonl` | 数据集路径 |
| `BATCH_SIZE` | `2` | 批大小 |
| `EPOCHS` | `3` | 训练轮数 |
| `LR` | `1e-4` | 学习率 |
| `MAX_LEN` | `512` | 最大序列长度 |

### 推理参数

| 参数 | 说明 |
|------|------|
| `temperature` | 控制随机性（0=确定，1=随机） |
| `top_p` | 核采样概率阈值 |
| `max_new_tokens` | 最大生成长度 |

---

## 📦 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+（GPU 训练推荐）

---

## 🔧 常见问题

**Q: 显存不足？**
- 减小 `BATCH_SIZE`
- 启用量化：`python train.py` 已内置 QLoRA

**Q: 训练很慢？**
- 确保使用 GPU
- 减少 `MAX_LEN`

**Q: 界面无法访问？**
- 检查是否运行在正确的端口
- 尝试 http://localhost:7860

---

## 📄 许可证

MIT License
