# inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 模型路径
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
LORA_PATH = "./qwen-lora"

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()  # 合并权重，加速推理（可选）
model.eval()

print("模型加载完成，开始对话（输入 'exit' 退出）\n")

def chat(messages):
    # 构建 prompt（Qwen3.5 格式）
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# 对话循环
messages = [{"role": "system", "content": "你是资深 Python 专家"}]

while True:
    user_input = input("用户: ")
    if user_input.lower() == "exit":
        break
    
    messages.append({"role": "user", "content": user_input})
    
    response = chat(messages)
    print(f"助手: {response}\n")
    
    messages.append({"role": "assistant", "content": response})