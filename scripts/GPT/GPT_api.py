import argparse
import json
import os
from tqdm import tqdm
from openai import OpenAI

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def summarize_text(client, document, model="gpt-4o-mini"):
    try:
        prompt = f"You are an expert at summarization. Summarize the following text: \n\n{document}\n\nSUMMARY:"
        
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

def process_documents(client, input_file, output_file):
    data = load_data(input_file)
    
    for entry in tqdm(data, desc="Processing Documents"):
        document = entry.get('document')
        if document:
            summary, prompt = summarize_text(client, document)  
            entry['prediction'] = summary  
            entry['used_prompt'] = prompt  
    
    save_data(data, output_file)

def main():
    parser = argparse.ArgumentParser(description="Summarize documents using OpenAI's GPT model.")
    
    parser.add_argument(
        "--api_key",
        required=True,
        help="API key for OpenAI access"
    )
    
    parser.add_argument(
        "--input_file_path",
        default="../data/style_shift_sampled/test/washington_test_sampled.json",
        help="Path to the input JSON file "
    )
    
    parser.add_argument(
        "--output_file_path",
        default="./output/style/output-gpt-washington_test_sampled.json",
        help="Path to the output JSON file "
    )
    
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    process_documents(client, args.input_file_path, args.output_file_path)

if __name__ == "__main__":
    main()
