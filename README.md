# Qwen QLoRA 微调模板

## 使用步骤

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

# 进入下载目录（假设文件在 Downloads）
cd C:\Users\26401\Downloads

# 安装
python -m pip install torch-2.6.0+cu124-cp312-cp312-win_amd64.whl torchvision-0.21.0+cu124-cp312-cp312-win_amd64.whl torchaudio-2.6.0+cu124-cp312-cp312-win_amd64.whl --no-cache-dir

python -m pip install datasets transformers peft accelerate bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir


python train.py
python infer.py
```

## 默认模型
- Qwen/Qwen2.5-0.5B-Instruct

## 输出目录
- `qwen-lora/`
