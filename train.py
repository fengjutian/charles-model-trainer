"""
Qwen3.5-0.8B 模型 LoRA 微调训练脚本
用于在本地数据集上对 Qwen 模型进行参数高效微调
"""

import os
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    Trainer
)
from peft import LoraConfig, get_peft_model

# ============================================================
# 配置区域
# ============================================================

# 模型名称或路径 (使用 Qwen 官方小模型)
MODEL_NAME = "Qwen/Qwen3.5-0.8B"

# ============================================================
# 1. 加载数据集
# ============================================================
# 从 JSONL 文件加载训练数据，期望格式为:
# {"messages": [{"role": "system/user/assistant", "content": "..."}]}

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# ============================================================
# 2. 初始化分词器
# ============================================================
# 设置 pad_token 为 eos_token，避免训练时出现 padding 相关警告

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 3. 4-bit 量化配置 (节省显存)
# ============================================================
# - load_in_4bit: 启用 4-bit 量化
# - bnb_4bit_compute_dtype: 计算时使用 float16
# - bnb_4bit_use_double_quant: 双重量化，进一步压缩
# - bnb_4bit_quant_type: 使用 NF4 量化类型 (适合神经网络权重)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ============================================================
# 4. 加载量化模型
# ============================================================
# device_map="auto": 自动将模型层分配到可用设备 (GPU/CPU)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)

# ============================================================
# 5. 数据预处理函数
# ============================================================
def tokenize_function(examples):
    """
    将对话格式的 messages 转换为模型可接受的文本格式
    
    格式说明:
    - system: 系统提示词
    - user: 用户输入
    - assistant: 助手回复
    
    每个角色使用 <|im_start|> 和 <|im_end|> 特殊标记包裹
    """
    texts = []
    for messages in examples["messages"]:
        text = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        texts.append(text)
    # 统一 token 序列长度到 max_length=256
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

# 对数据集进行 tokenize 处理，并移除原始 messages 列
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["messages"])

# ============================================================
# 6. LoRA 配置
# ============================================================
# LoRA 参数说明:
# - r (rank): LoRA 矩阵的秩，越大效果越好但参数量越多
# - lora_alpha: 缩放因子，通常设为 r 的 2 倍
# - lora_dropout: Dropout 概率，防止过拟合
# - target_modules: 要应用 LoRA 的模块 (Qwen 的注意力层和 FFN 层)

lora_config = LoraConfig(
    r=16,                                     # LoRA 秩
    lora_alpha=32,                            # 缩放因子
    lora_dropout=0.05,                        # Dropout 概率
    task_type="CAUSAL_LM",                    # 因果语言模型任务
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

# ============================================================
# 7. 应用 LoRA 到模型
# ============================================================
# 使用 PEFT 库将 LoRA adapter 注入到原模型中

model = get_peft_model(model, lora_config)

# 启用梯度检查点 (Gradient Checkpointing)
# 用计算换显存：反向传播时不保存所有中间激活，而是重新计算
model.gradient_checkpointing_enable()

# 打印可训练参数统计 (LoRA 参数量 vs 总参数量)
model.print_trainable_parameters()

# ============================================================
# 8. 数据整理器
# ============================================================
# mlm=False: 这是因果语言模型，不是掩码语言模型
# 会在 batch 内部自动进行 padding 并创建 attention mask

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ============================================================
# 9. 训练参数配置
# ============================================================
args = TrainingArguments(
    output_dir="./qwen-lora",           # 模型保存目录
    per_device_train_batch_size=1,      # 每设备 batch 大小 (显存有限设小一点)
    gradient_accumulation_steps=32,     # 累积 32 步相当于 effective batch size = 32
    learning_rate=2e-4,                 # 学习率 (LoRA 通常用较大学习率)
    num_train_epochs=3,                 # 训练 3 个 epoch
    logging_steps=1,                    # 每 1 步打印日志
    save_steps=50,                      # 每 50 步保存 checkpoint
    fp16=True,                          # 启用混合精度训练 (加速 + 节省显存)
    optim="paged_adamw_8bit",           # 使用 8-bit 优化的 AdamW (节省显存)
    report_to="none",                   # 不连接 wandb/mlflow 等追踪工具
)

# ============================================================
# 10. 初始化 Trainer 并开始训练
# ============================================================

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# ============================================================
# 11. 保存微调后的 LoRA 权重
# ============================================================
model.save_pretrained("./qwen-lora")
print("Training complete. Model saved to ./qwen-lora")
