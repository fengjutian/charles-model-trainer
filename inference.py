# inference.py
# 微调模型推理脚本 - 加载 LoRA 权重进行对话

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# 配置区域
# ============================================================

# 基础模型名称或路径 (需与训练时一致)
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
# LoRA 权重保存路径 (训练 output_dir)
LORA_PATH = "./qwen-lora"

# ============================================================
# 1. 加载 Tokenizer
# ============================================================
# 分词器用于编码输入文本和解码生成文本

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# ============================================================
# 2. 加载基础模型 (FP16 精度)
# ============================================================
# 注意：推理时通常不用量化，保持 FP16 精度以获得更好效果

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# ============================================================
# 3. 加载并合并 LoRA 权重
# ============================================================
# PeftModel.from_pretrained: 加载 LoRA adapter 到基础模型
# merge_and_unload(): 将 LoRA 权重合并到基础模型，合并后推理更快

model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()  # 合并权重，加速推理（可选）

# 切换到评估模式，禁用 dropout
model.eval()

print("模型加载完成，开始对话（输入 'exit' 退出）\n")

# ============================================================
# 4. 对话处理函数
# ============================================================
def chat(messages):
    """
    将对话历史发送给模型并获取回复
    
    参数:
        messages: 包含对话历史的列表，格式如:
                 [{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."}]
    返回:
        模型生成的回复文本
    """
    # 使用 chat template 将 messages 转换为模型可识别的格式
    # tokenize=False: 返回字符串而非 token IDs
    # add_generation_prompt=True: 添加助手生成开始标记
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 将文本编码为 tensor，并移动到模型所在设备
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 推理时不需要计算梯度，使用 torch.no_grad() 节省显存
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,      # 最多生成 512 个新 token
            temperature=0.7,         # 控制随机性 (0=确定, 1=高随机)
            top_p=0.9,               # 核采样：只采样累计概率前 90% 的 token
            do_sample=True,          # 启用采样 (False 则 greedy 解码)
        )
    
    # 解码输出，跳过输入部分 (保留新生成的内容)
    # [inputs.input_ids.shape[1]:] 从新 token 开始处切片
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# ============================================================
# 5. 对话循环
# ============================================================

# 初始化系统提示词
messages = [{"role": "system", "content": "你是资深 Python 专家"}]

# 持续对话直到用户输入 'exit'
while True:
    user_input = input("用户: ")
    if user_input.lower() == "exit":
        break
    
    # 添加用户消息
    messages.append({"role": "user", "content": user_input})
    
    # 获取模型回复
    response = chat(messages)
    print(f"助手: {response}\n")
    
    # 将助手回复添加到历史，以便多轮对话
    messages.append({"role": "assistant", "content": response})