from huggingface_hub import login
import argparse
import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os

# Set up argument parser to handle input for model directory, dataset path, and Hugging Face token
parser = argparse.ArgumentParser(description="Fine-tune Llama model on a dataset.")
parser.add_argument('--new_model', type=str, required=True, help="Directory name to save the fine-tuned model")
parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset file (json format)")
parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face token for authentication")

args = parser.parse_args()

# Login to Hugging Face Hub with the provided token
login(token=args.hf_token)

# Specify the model ID of the pre-trained model to be fine-tuned
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Load model and tokenizer with quantization settings
def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype="float16", 
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache=False  # Disable cache for training
    model.config.pretraining_tp=1
    return model, tokenizer

model, tokenizer = get_model_and_tokenizer(model_id)
print('Model loaded successfully')

# Configure LoRA (Low-Rank Adaptation) to optimize specific layers during fine-tuning
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Display which parameters are trainable

# Load and prepare the dataset
dataset = load_dataset('json', data_files=args.dataset_path, split='all')
dataset = dataset.shuffle(seed=42).select(range(1000))  # Shuffle and sample 1000 examples for training

# Define a function to format data into a template for the model
def format_chat_template(row):
    article = row["document"]
    prompt = f"You are an expert at summarization. Summarize the following text: \n\n{article}\n\nSUMMARY:"
    row_json = [{"role": "user", "content": prompt},
               {"role": "assistant", "content": row["summary"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

# Apply formatting function to dataset
dataset = dataset.map(
    format_chat_template,
    num_proc=4 
)
print(f"Number of samples in the dataset: {len(dataset)}")

# Set up training configurations
training_arguments = TrainingArguments(
    output_dir=args.new_model,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    num_train_epochs=20,
    logging_steps=0.05,
    warmup_steps=0,
    logging_strategy="steps",
    learning_rate=5e-4,
    fp16=False,
    bf16=True,  # Mixed-precision training for faster performance on supported hardware
    group_by_length=True,
    save_strategy="steps",  
    save_steps=0.05, 
    save_total_limit=1
)

# Initialize Trainer for fine-tuning with custom model and dataset configurations
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()  # Start training process

model.config.use_cache = True  # Re-enable cache post-training
