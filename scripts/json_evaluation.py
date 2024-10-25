import json
from evaluate import load
from tqdm import tqdm
import math

def main():
    json_files = [
         './output/genre/2shot/output-llama3.1-8b-2shot-cnndm_test_sampled_500.json',
         './output/genre/2shot/output-llama3.1-8b-2shot-pubmed_test_sampled_500.json',
        # './output/genre/2shot/output-llama3.1-8b-2shot-reddit_test_sampled_500.json',
        # './output/genre/2shot/output-llama3.1-8b-2shot-samsum_test_sampled_500.json',
        # './output/genre/2shot/output-llama3.1-8b-2shot-wikihow_test_sampled_500.json',
        # './output/style/2shot/output-llama3.1-8b-2shot-cnn_test_sampled_500.json',
        # './output/style/2shot/output-llama3.1-8b-2shot-fox_test_sampled_500.json',
        # './output/style/2shot/output-llama3.1-8b-2shot-nydaily_test_sampled_500.json',
        # './output/style/2shot/output-llama3.1-8b-2shot-nytimes_test_sampled_500.json',
        # './output/style/2shot/output-llama3.1-8b-2shot-washington_test_sampled_500.json',
        # './output-70B/topic/output-llama3.1-70b-topic1_test_sampled.json',
        # './output/topic/2shot/output-llama3.1-8b-2shot-topic2_test_sampled_500.json',
        # './output/topic/2shot/output-llama3.1-8b-2shot-topic3_test_sampled_500.json',
        # './output/topic/2shot/output-llama3.1-8b-2shot-topic4_test_sampled_500.json',
        # './output/topic/2shot/output-llama3.1-8b-2shot-topic5_test_sampled_500.json',
    ]

    rouge = load('rouge')
    bertscore = load('bertscore')

    for json_file in tqdm(json_files, desc="Processing JSON files"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        references = []
        predictions = []

        for item in tqdm(data, desc=f"Extracting data from {json_file}", leave=False):
            references.append(item['summary'])
            predictions.append(item['prediction'])

        # ROUGE
        rouge_scores = rouge.compute(predictions=predictions, references=references)

        
        rouge_sum = (rouge_scores['rouge1'] + 
                     rouge_scores['rouge2'] + 
                     rouge_scores['rougeL']) / 3

        # BERTScore
        batch_size = 32
        num_samples = len(predictions)
        num_batches = math.ceil(num_samples / batch_size)

        bert_precision = []
        bert_recall = []
        bert_f1 = []

        for i in tqdm(range(num_batches), desc=f"Computing BERTScore for {json_file}", leave=False):
            batch_predictions = predictions[i*batch_size:(i+1)*batch_size]
            batch_references = references[i*batch_size:(i+1)*batch_size]
            bert_scores = bertscore.compute(predictions=batch_predictions, references=batch_references, lang='en')
            bert_precision.extend(bert_scores['precision'])
            bert_recall.extend(bert_scores['recall'])
            bert_f1.extend(bert_scores['f1'])

        avg_precision = sum(bert_precision) / len(bert_precision)
        avg_recall = sum(bert_recall) / len(bert_recall)
        avg_f1 = sum(bert_f1) / len(bert_f1)


        evaluation_results = {
            'ROUGE': {
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeL': rouge_scores['rougeL'],
                'rougeSUM': rouge_sum
            },
            'BERTScore': {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
            }
        }

        output_json_file = f"./evaluation{json_file.replace('./output', '').replace('.json', '')}_evaluation.json"

        with open(output_json_file, 'w', encoding='utf-8') as outfile:
            json.dump(evaluation_results, outfile, ensure_ascii=False, indent=4)

        print(f"Evaluation results saved to {output_json_file}\n")

if __name__ == '__main__':
    main()
