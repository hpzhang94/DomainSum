import os
import json
import random

# Define a function to read and process a single JSON file
def process_json_file(file_path, seed=65, sample_size=1000):
    new_data = []
    
    # Open and read the entire JSON file (each file contains multiple JSON objects)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # If the number of data entries exceeds sample_size, use a random seed to select sample_size entries
    if len(data) > sample_size:
        random.seed(seed)
        data = random.sample(data, sample_size)
    
    # Extract "document" and "summary" keys from each entry
    for entry in data:
        if 'document' in entry and 'summary' in entry:
            new_entry = {
                "document": entry["document"],
                "summary": entry["summary"]
            }
            new_data.append(new_entry)
    
    return new_data

# Process a single JSON file and save the new JSON file
def process_and_save_json_file(input_file_path, output_file_path, seed=65, sample_size=10000):
    # Randomly sample data from the input file
    sampled_data = process_json_file(input_file_path, seed=seed, sample_size=sample_size)
    
    # Open the new file to save the processed data
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in sampled_data:
            # Write each entry to the file as an individual JSON object, not in a list
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")  # Add a newline to separate each JSON object
    print(f"Processed and saved: {output_file_path}")

# Process all JSON files in the specified folder
def process_all_json_files(folder_path, output_folder, seed=65, sample_size=10000):
    # Get all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # Process each JSON file one by one
    for json_file in json_files:
        input_file_path = os.path.join(folder_path, json_file)
        output_file_path = os.path.join(output_folder, json_file.replace('.json', '_llm.json'))
        
        # Process the file and save the result
        process_and_save_json_file(input_file_path, output_file_path, seed=seed, sample_size=sample_size)

if __name__ == "__main__":
    folder_path = "./data/train" 
    output_folder = "./data/train_processed_10000" 
    process_all_json_files(folder_path, output_folder)
