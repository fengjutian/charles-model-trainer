from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "./qwen-lora")

prompt = "请解释 Python 中的装饰器。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
