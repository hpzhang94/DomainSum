import argparse
import json
import os
from tqdm import tqdm
import random
from openai import OpenAI

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def select_random_examples(extra_data):
    # Select two random examples from extra data
    examples = random.sample(extra_data, 2)
    example1_document = examples[0].get('document', '')
    example1_summary = examples[0].get('summary', '')
    example2_document = examples[1].get('document', '')
    example2_summary = examples[1].get('summary', '')
    
    return example1_document, example1_summary, example2_document, example2_summary

def summarize_text(client, document, extra_data, model="gpt-4o-mini"):
    try:
        # Select Example1 and Example2 randomly
        example1_document, example1_summary, example2_document, example2_summary = select_random_examples(extra_data)

        prompt = f'''
            You are an expert at summarization. Here are two examples of how to summarize a text:

            Example 1:
            Document: {example1_document}
            Summary: {example1_summary}

            Example 2:
            Document: {example2_document}
            Summary: {example2_summary}

            Now, summarize the following text:

            Document: {document}

            Summary:
            '''
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at summarization."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  
            max_tokens=100
        )
        
        summary = response.choices[0].message.content.strip()
        
        return summary, prompt  
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "", ""

def save_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def process_documents(client, input_file, extra_file, output_file):
    data = load_data(input_file)
    extra_data = load_data(extra_file)

    for entry in tqdm(data, desc="Processing Documents"):
        document = entry.get('document')
        if document:
            summary, prompt = summarize_text(client, document, extra_data)
            entry['prediction'] = summary
            entry['used_prompt'] = prompt
    
    save_data(data, output_file)

def main():
    parser = argparse.ArgumentParser(description="Summarize documents using OpenAI's GPT model with examples.")
    
    parser.add_argument(
        "--api_key",
        required=True,
        help="API key for OpenAI access"
    )
    
    parser.add_argument(
        "--input_file_path",
        default = '../data/topic_shift_sampled/test/Soccer_test_sampled.json',
        help="Path to the input JSON file"
    )
    
    parser.add_argument(
        "--extra_file_path",
        default = '../data/topic_shift_sampled/train/Soccer_train_sampled_20.json',
        help="Path to the extra JSON file with example summaries"
    )
    
    parser.add_argument(
        "--output_file_path",
        default =  './output/topic/2shot/output-gpt2shot-Soccer_test_sampled.json',
        help="Path to the output JSON file"
    )
    
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    process_documents(client, args.input_file_path, args.extra_file_path, args.output_file_path)

if __name__ == "__main__":
    main()
