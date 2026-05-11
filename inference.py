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
def chat(messages, temperature=0.7, top_p=0.9, max_new_tokens=512, do_sample=True):
    """
    将对话历史发送给模型并获取回复
    
    参数:
        messages: 包含对话历史的列表，格式如:
                 [{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."}]
        temperature: 控制随机性 (0=确定, 1=高随机)
        top_p: 核采样：只采样累计概率前 X% 的 token
        max_new_tokens: 最多生成的新 token 数量
        do_sample: 是否启用采样 (False 则 greedy 解码)
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
    
    # 解码输出，跳过输入部分 (保留新生成的内容)
    # [inputs.input_ids.shape[1]:] 从新 token 开始处切片
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# ============================================================
# 5. 快捷命令与提问方式
# ============================================================

def show_help():
    """显示帮助信息"""
    print("\n📖 快捷命令:")
    print("  help     - 显示此帮助信息")
    print("  clear    - 清空对话历史")
    print("  history  - 显示对话历史")
    print("  save     - 保存对话到文件")
    print("  load     - 加载对话文件")
    print("  system   - 修改系统提示词")
    print("  config   - 调整生成参数")
    print("  tokens   - 查看当前上下文长度")
    print("  exit     - 退出程序")
    print()

def clear_history():
    """清空对话历史"""
    global messages
    messages = [{"role": "system", "content": "你是资深 Python 专家"}]
    print("✅ 对话历史已清空\n")

def show_history():
    """显示对话历史"""
    print("\n📝 对话历史:")
    for i, msg in enumerate(messages):
        role = {"system": "系统", "user": "用户", "assistant": "助手"}[msg["role"]]
        content = msg["content"][:100] + ("..." if len(msg["content"]) > 100 else "")
        print(f"  [{i+1}] {role}: {content}")
    print()

def save_conversation(filepath="conversation.json"):
    """保存对话历史"""
    import json
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"✅ 对话已保存到 {filepath}\n")

def load_conversation(filepath="conversation.json"):
    """加载对话历史"""
    import json
    global messages
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            messages = json.load(f)
        print(f"✅ 对话已从 {filepath} 加载\n")
    except FileNotFoundError:
        print(f"❌ 文件 {filepath} 不存在\n")

def change_system_prompt():
    """修改系统提示词"""
    global messages
    print("\n当前系统提示词:", messages[0]["content"])
    new_prompt = input("请输入新的系统提示词: ").strip()
    if new_prompt:
        messages[0] = {"role": "system", "content": new_prompt}
        print("✅ 系统提示词已更新\n")
    else:
        print("⚠️ 输入为空，未修改\n")

def show_config():
    """显示当前生成参数"""
    print("\n⚙️ 当前生成参数:")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  do_sample: {do_sample}")
    print()

def adjust_config():
    """调整生成参数"""
    global temperature, top_p, max_new_tokens, do_sample
    
    print("\n🔧 调整参数 (直接回车保持当前值):")
    
    new_temp = input(f"  temperature (当前: {temperature}): ").strip()
    if new_temp:
        temperature = float(new_temp)
    
    new_top_p = input(f"  top_p (当前: {top_p}): ").strip()
    if new_top_p:
        top_p = float(new_top_p)
    
    new_max = input(f"  max_new_tokens (当前: {max_new_tokens}): ").strip()
    if new_max:
        max_new_tokens = int(new_max)
    
    new_sample = input(f"  do_sample (当前: {do_sample}, True/False): ").strip()
    if new_sample.lower() == "true":
        do_sample = True
    elif new_sample.lower() == "false":
        do_sample = False
    
    print("✅ 参数已更新\n")

def show_token_count():
    """显示当前上下文长度"""
    total = sum(len(tokenizer.encode(msg["content"])) for msg in messages)
    print(f"\n📊 当前上下文 token 数量: ~{total}\n")

# ============================================================
# 6. 对话循环
# ============================================================

# 初始化系统提示词
messages = [{"role": "system", "content": "你是资深 Python 专家"}]

# 初始化生成参数（可运行时调整）
temperature = 0.7
top_p = 0.9
max_new_tokens = 512
do_sample = True

print("🎯 智能助手已就绪")
print("💡 输入 'help' 查看所有快捷命令\n")

# 持续对话直到用户输入 'exit'
while True:
    user_input = input("👤 您: ").strip()
    
    # 处理快捷命令
    if user_input.lower() in ["help", "h", "？"]:
        show_help()
        continue
    elif user_input.lower() in ["clear", "c", "清空"]:
        clear_history()
        continue
    elif user_input.lower() in ["history", "hist", "历史"]:
        show_history()
        continue
    elif user_input.lower() in ["save", "s"]:
        save_conversation()
        continue
    elif user_input.lower() in ["load", "l"]:
        filename = input("请输入文件名 (默认: conversation.json): ").strip()
        load_conversation(filename if filename else "conversation.json")
        continue
    elif user_input.lower() in ["system", "sys"]:
        change_system_prompt()
        continue
    elif user_input.lower() in ["config", "cfg"]:
        show_config()
        adjust_config()
        continue
    elif user_input.lower() in ["tokens", "token"]:
        show_token_count()
        continue
    elif user_input.lower() in ["exit", "quit", "q"]:
        print("\n👋 再见!")
        break
    elif not user_input:
        print("⚠️ 请输入内容\n")
        continue
    
    # 添加用户消息
    messages.append({"role": "user", "content": user_input})
    
    # 获取模型回复
    print("🤖 助手: ", end="", flush=True)
    response = chat(messages)
    print(f"{response}\n")
    
    # 将助手回复添加到历史，以便多轮对话
    messages.append({"role": "assistant", "content": response})