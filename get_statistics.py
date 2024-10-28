import json
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from typing import List, Dict, Tuple
import numpy as np
import os
from tqdm import tqdm  # Display Progress Bar

# Ensure the nltk punkt tokenizer has been downloaded
nltk.download('punkt')

def load_json(file_path):
    """
    Load JSON data from a specified file path.
    """
    print(f"Loading JSON file from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} documents from {file_path}")
    return data

def calculate_novel_ngrams(documents, summaries):
    """
    Calculate the average novel 1-grams, 2-grams, and 3-grams for all documents and summaries.
    """
    novel_ngrams_counts = {1: [], 2: [], 3: []}

    print("Calculating novel n-grams...")
    for document, summary in tqdm(zip(documents, summaries), total=len(documents), desc="Processing documents"):
        document_tokens = word_tokenize(document.lower())
        summary_tokens = word_tokenize(summary.lower())

        for n in range(1, 4):
            document_ngrams = set(ngrams(document_tokens, n))
            summary_ngrams = set(ngrams(summary_tokens, n))
            novel_ngrams = len(summary_ngrams - document_ngrams)
            novel_ngrams_counts[n].append(novel_ngrams / len(summary_ngrams) if len(summary_ngrams) > 0 else 0)

    # Calculate averages
    avg_novel_ngrams = {f'novel_{n}_grams_avg': np.mean(novel_ngrams_counts[n]) for n in novel_ngrams_counts}
    print("Finished calculating novel n-grams.")
    return avg_novel_ngrams

def calculate_token_statistics(tokens_list):
    """
    Calculate the average token length and standard deviation for documents or summaries.
    """
    print("Calculating token statistics...")
    token_lengths = [len(tokens) for tokens in tokens_list]
    avg_length = np.mean(token_lengths)
    std_dev_length = np.std(token_lengths)
    print("Finished calculating token statistics.")
    return avg_length, std_dev_length

def calculate_density(document: str, summary: str) -> float:
    """
    Calculate the density for a given document and summary.
    """
    document_tokens = word_tokenize(document.lower())
    summary_tokens = word_tokenize(summary.lower())
    
    extractive_fragments = []
    i, j = 0, 0
    while i < len(document_tokens) and j < len(summary_tokens):
        if document_tokens[i] == summary_tokens[j]:
            frag_len = 0
            while i + frag_len < len(document_tokens) and j + frag_len < len(summary_tokens) and document_tokens[i + frag_len] == summary_tokens[j + frag_len]:
                frag_len += 1
            if frag_len > 0:
                extractive_fragments.append(frag_len)
            i += frag_len
            j += frag_len
        else:
            i += 1

    density = sum(f ** 2 for f in extractive_fragments) / len(summary_tokens) if len(summary_tokens) > 0 else 0
    return density

def calculate_compression(document: str, summary: str) -> float:
    """
    Calculate the compression ratio for a given document and summary.
    """
    document_tokens = word_tokenize(document.lower())
    summary_tokens = word_tokenize(summary.lower())
    
    compression = len(document_tokens) / len(summary_tokens) if len(summary_tokens) > 0 else 0
    return compression

def calculate_coverage(summary_ngrams, document_ngrams):
    """
    Calculate if the n-grams in the summary appear in the document.
    """
    overlap = sum(1 for ngram in summary_ngrams if ngram in document_ngrams)
    return overlap / len(summary_ngrams) if summary_ngrams else 0

def calculate_average_coverage(summary_text, document_text):
    """
    Calculate the average coverage for 1-gram, 2-gram, and 3-gram.
    """
    total_coverage = 0
    for n in range(1, 4):
        summary_ngrams = set(ngrams(word_tokenize(summary_text.lower()), n))
        document_ngrams = set(ngrams(word_tokenize(document_text.lower()), n))
        total_coverage += calculate_coverage(summary_ngrams, document_ngrams)
    return total_coverage / 3

def calculate_diversity(ngrams_list):
    """
    Calculate the diversity of n-grams.
    """
    unique_ngrams = set(ngrams_list)
    return len(unique_ngrams) / len(ngrams_list) if ngrams_list else 0

def calculate_average_diversity(text):
    """
    Calculate the average diversity for 1-gram, 2-gram, and 3-gram.
    """
    total_diversity = 0
    for n in range(1, 4):
        ngrams_list = list(ngrams(word_tokenize(text.lower()), n))
        total_diversity += calculate_diversity(ngrams_list)
    return total_diversity / 3

def process_json_files(json_files):
    """
    Process a list of JSON file paths, calculate statistics for each file, and save results to statistics.txt.
    """
    results = []

    with open('./statistics_with_diversity.txt', 'w') as stats_file:
        for json_file_path in json_files:
            data = load_json(json_file_path)
            documents = [item['document'] for item in data]
            summaries = [item['summary'] for item in data]

            # Calculate average novel n-grams
            avg_novel_ngrams = calculate_novel_ngrams(documents, summaries)
            
            # Calculate average token length and standard deviation for all documents
            document_tokens_list = [word_tokenize(doc.lower()) for doc in documents]
            avg_doc_length, std_doc_length = calculate_token_statistics(document_tokens_list)
            
            # Calculate average token length and standard deviation for all summaries
            summary_tokens_list = [word_tokenize(summary.lower()) for summary in summaries]
            avg_summary_length, std_summary_length = calculate_token_statistics(summary_tokens_list)

            # Calculate density and compression ratio
            densities = [calculate_density(doc, summary) for doc, summary in zip(documents, summaries)]
            avg_density = np.mean(densities)
            
            compressions = [calculate_compression(doc, summary) for doc, summary in zip(documents, summaries)]
            avg_compression = np.mean(compressions)

            # Calculate coverage
            coverages = [calculate_average_coverage(summary, doc) for doc, summary in zip(documents, summaries)]
            avg_coverage = np.mean(coverages)

            # Calculate diversity
            document_diversity = [calculate_average_diversity(doc) for doc in documents]
            summary_diversity = [calculate_average_diversity(summary) for summary in summaries]
            avg_document_diversity = np.mean(document_diversity)
            avg_summary_diversity = np.mean(summary_diversity)

            # Calculate abstractiveness (average of novel 3-grams)
            abstractiveness = avg_novel_ngrams['novel_3_grams_avg']

            # Print and write to file
            stats_file.write(f"File: {json_file_path}\n")
            stats_file.write(f"Total number of documents: {len(documents)}\n")
            stats_file.write(f"Average novel n-grams: {avg_novel_ngrams}\n")
            stats_file.write(f"Average document length: {avg_doc_length:.2f} tokens (std: {std_doc_length:.2f})\n")
            stats_file.write(f"Average summary length: {avg_summary_length:.2f} tokens (std: {std_summary_length:.2f})\n")
            stats_file.write(f"Average density: {avg_density:.4f}\n")
            stats_file.write(f"Average compression: {avg_compression:.4f}\n")
            stats_file.write(f"Average coverage: {avg_coverage:.4f}\n")
            stats_file.write(f"Average document diversity: {avg_document_diversity:.4f}\n")
            stats_file.write(f"Average summary diversity: {avg_summary_diversity:.4f}\n")
            stats_file.write(f"Abstractiveness: {abstractiveness:.4f}\n\n")

            results.append([
                os.path.basename(json_file_path), str(len(documents)), 
                f"{avg_novel_ngrams['novel_1_grams_avg']:.2f}",
                f"{avg_novel_ngrams['novel_2_grams_avg']:.2f}",
                f"{avg_novel_ngrams['novel_3_grams_avg']:.2f}",
                f"{avg_doc_length:.2f}", f"{std_doc_length:.2f}",
                f"{avg_summary_length:.2f}", f"{std_summary_length:.2f}",
                f"{avg_density:.4f}", f"{avg_compression:.4f}",
                f"{avg_coverage:.4f}", f"{avg_document_diversity:.4f}", f"{avg_summary_diversity:.4f}",
                f"{abstractiveness:.4f}"
            ])

json_files_list = ['./cnndm_test.json', './pubmed_test.json', './reddit_test.json', './samsum_test.json', './wikihow_test.json']
process_json_files(json_files_list)
