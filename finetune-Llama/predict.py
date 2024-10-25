import argparse
from huggingface_hub import login
from tqdm import tqdm
import json
import torch
from datasets import load_dataset, Dataset
from peft import (
    PeftModel,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline
from trl import setup_chat_format
import os
import pandas as pd

# Define the argument parser
parser = argparse.ArgumentParser(description="Fine-tune LLaMA model with a custom dataset.")
parser.add_argument('--new_model', type=str, default = "./llama-3.1-8b-topic1/checkpoint-3750", help="Path to the fine-tuned model")
parser.add_argument('--input_json_path', type=str, default = "./data/test_processed/cnndm_test_sampled_llm.json", help="Path to the input JSON file")
parser.add_argument('--output_json_path', type=str, required=True, help="Path to the output JSON file")
parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face token for authentication")

args = parser.parse_args()

# Login to Hugging Face Hub with the provided token
login(token=args.hf_token)

# Base model and tokenizer
base_model = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='left')

base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)

# Load the fine-tuned model
model = PeftModel.from_pretrained(base_model_reload, args.new_model)

# Merge adapter with base model
model = model.merge_and_unload()

# Setup the pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    batch_size=6
)

def extract_assistant_response(text):
    text = ' '.join(text.split())
    start_marker = "<|im_start|>assistant"
    start_index = text.find(start_marker)
    if start_index == -1:
        return "Assistant's response not found."
    start_index += len(start_marker)
    return text[start_index:].strip()

def load_articles_from_json(file_path, num_articles=500):
    articles = []
    count = 0  
    with open(file_path, 'r') as file:
        for line in file:
            if count >= num_articles:  
                break
            try:
                article = json.loads(line)
                articles.append(article)
                count += 1 
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
    return Dataset.from_pandas(pd.DataFrame(articles))

def generate_and_save_responses(input_file_path, output_file_path):
    dataset = load_articles_from_json(input_file_path)
    results = []
    
    def process_batch(batch):
        prompts = []
        for article in batch['document']:
            article_prompt = f"You are an expert at summarization. Summarize the following text: \n\n{article}\n\nSUMMARY:"
            messages = [{"role": "user", "content": article_prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
        outputs = pipe(prompts, max_new_tokens=100, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)
        
        predictions = []
        for sublist in outputs:
            for output in sublist:
                if 'generated_text' in output:
                    extracted_response = extract_assistant_response(output['generated_text'])
                    predictions.append(extracted_response)
                else:
                    predictions.append("Generated text not found")
        
        return {"prediction": predictions}
    
    dataset = dataset.map(process_batch, batched=True, batch_size=4)
    results = [{
        "document": item['document'],
        "summary": item['summary'],
        "prediction": item['prediction']
    } for item in dataset]
    
    # Save results to JSON file
    with open(output_file_path, 'w') as outfile:
        json.dump(results, outfile, indent=6)

# Call the function with command line arguments
generate_and_save_responses(args.input_json_path, args.output_json_path)
