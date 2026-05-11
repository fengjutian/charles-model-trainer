# Gradio 前端页面 - Web 界面进行对话

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# 配置区域
# ============================================================

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
LORA_PATH = "./qwen-lora"

# ============================================================
# 模型加载
# ============================================================

print("正在加载模型...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()
model.eval()

print("模型加载完成!")

# ============================================================
# 对话函数
# ============================================================

def chat_with_model(messages, temperature, top_p, max_tokens):
    """
    与模型对话
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    return response.strip()

# ============================================================
# Gradio 界面
# ============================================================

def respond(message, history, system_prompt, temperature, top_p, max_tokens):
    """
    处理用户输入并返回回复
    """
    if not message.strip():
        return history, ""
    
    # 构建消息列表
    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加历史对话（兼容元组和字典格式）
    for item in history:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            messages.append({"role": "user", "content": item[0]})
            messages.append({"role": "assistant", "content": item[1]})
    
    # 添加当前消息
    messages.append({"role": "user", "content": message})
    
    # 获取回复
    response = chat_with_model(messages, temperature, top_p, max_tokens)
    
    # 更新对话历史（使用元组格式）
    history.append((message, response))
    return history, ""

# ============================================================
# 启动 Gradio
# ============================================================

# 创建界面
demo = gr.Blocks(title="Qwen 智能助手")

with demo:
    gr.Markdown("# 🤖 Qwen 智能助手")
    gr.Markdown("基于 Qwen3.5-0.8B 微调模型的对话系统")
    
    with gr.Row():
        with gr.Column(scale=3):
            # 聊天机器人组件
            chatbot = gr.Chatbot(height=500)
            
            # 输入框
            msg = gr.Textbox(
                label="输入您的问题",
                placeholder="请输入问题，按 Enter 发送...",
                lines=3
            )
            
            # 提交按钮
            with gr.Row():
                submit_btn = gr.Button("发送 🚀", variant="primary")
                clear_btn = gr.Button("清空对话", variant="secondary")
        
        with gr.Column(scale=1):
            # 参数设置面板
            gr.Markdown("### ⚙️ 参数设置")
            
            system_prompt = gr.Textbox(
                label="系统提示词",
                value="你是资深 Python 专家",
                lines=2
            )
            
            gr.Markdown("**生成参数**")
            temperature = gr.Slider(
                minimum=0, maximum=2, value=0.7, step=0.1,
                label="temperature（随机性）"
            )
            top_p = gr.Slider(
                minimum=0, maximum=1, value=0.9, step=0.05,
                label="top_p（核采样）"
            )
            max_tokens = gr.Slider(
                minimum=64, maximum=2048, value=512, step=64,
                label="max_tokens（最大长度）"
            )
            
            gr.Markdown("---")
            gr.Markdown("### 💡 使用说明")
            gr.Markdown("""
            1. 在输入框中输入问题
            2. 点击发送或按 Enter 提交
            3. 调整右侧参数改变生成效果
            4. 点击清空对话重置聊天
            """)
    
    # 事件绑定
    submit_btn.click(respond, inputs=[msg, chatbot, system_prompt, temperature, top_p, max_tokens], outputs=[chatbot, msg])
    msg.submit(respond, inputs=[msg, chatbot, system_prompt, temperature, top_p, max_tokens], outputs=[chatbot, msg])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

# ============================================================
# 启动应用
# ============================================================

if __name__ == "__main__":
    import traceback
    
    try:
        print("\n🌐 启动 Web 界面...")
        print("📍 访问地址: http://127.0.0.1:7860")
        print("📍 或者访问: http://localhost:7860")
        print("\n按 Ctrl+C 停止服务器\n")
        
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
    except Exception as e:
        print("\n❌ 发生错误:")
        traceback.print_exc()
        input("\n按 Enter 退出...")