import os
import json
import time
import argparse
from tqdm import tqdm
from huggingface_hub import InferenceClient

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def summarize_text(document, client):
    try:
        # Zero-shot prompt
        prompt = f"You are an expert at summarization. Summarize the following text: \n\n{document}\n\nSUMMARY:"
        
        summary = ""
        for message in client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
            stream=True,
        ):
            summary += message.choices[0].delta.content
        
        return summary.strip(), prompt
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "", ""

def save_data(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the output folder is created
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def process_documents(input_file, output_file, client):
    data = load_data(input_file)

    for entry in tqdm(data, desc="Processing Documents"):
        document = entry.get('document')
        if document:
            summary, prompt = summarize_text(document, client)
            entry['prediction'] = summary
            entry['used_prompt'] = prompt
    
    save_data(data, output_file)

def delay_process(minutes, input_file_path, output_file_path, model_name, token):
    delay_seconds = minutes * 60
    print(f"Process will start after {minutes} minute(s).")
    time.sleep(delay_seconds)

    # Initialize the InferenceClient directly using the provided model name
    client = InferenceClient(model_name, token=token)

    process_documents(input_file_path, output_file_path, client)

# Main function for argument parsing and folder setup
def main():
    parser = argparse.ArgumentParser(description="Process some documents.")
    parser.add_argument('--model', type=str, help='Model to use for summarization (e.g., mistralai/Mistral-7B-Instruct)')
    parser.add_argument('--input', type=str, help='Path to the input file')
    parser.add_argument('--delay', type=int, default=0, help='Delay in minutes before starting the process')
    parser.add_argument('--token', type=str, required=True, help='API token for InferenceClient')
    parser.add_argument('--output', type=str, help='Optional custom output directory')

    args = parser.parse_args()

    # Get the model name directly from the input
    model_name = args.model

    # Check if a custom output directory is provided
    if args.output:
        # Use the provided output directory
        output_folder = os.path.join(args.output)
    else:
        # Use the default output directory
        output_folder = os.path.join('./output_0shot/')
    
    os.makedirs(output_folder, exist_ok=True)

    # Create the output file path dynamically based on the model name, replace \ with -
    output_file = os.path.join(output_folder, f"{model_name}_zeroshot_prediction_result.json").replace("\\", "-")

    print(f"Process will start after {args.delay} minute(s).")

    print(f"Output will be saved to: {output_file}")

    # Call the delay process function
    delay_process(args.delay, args.input, output_file, model_name, args.token)

if __name__ == '__main__':
    main()
