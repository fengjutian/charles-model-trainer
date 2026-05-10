from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

args = TrainingArguments(
    output_dir="./qwen-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    save_steps=50,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=args,
)

trainer.train()
trainer.save_model("./qwen-lora")
print("Training complete. Model saved to ./qwen-lora")
