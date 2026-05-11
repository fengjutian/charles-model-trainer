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

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)

def tokenize_function(examples):
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
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["messages"])

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="./qwen-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    save_steps=50,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./qwen-lora")
print("Training complete. Model saved to ./qwen-lora")
