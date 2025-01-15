# DomainSum

## Introduction

DomainSum is a hierarchical benchmark for evaluating the impact of domain shifts on abstractive summarization models.
- Includes five distinct domains at each level (genre, style, and topic).
- Constructed from high-quality public datasets.
- Eight key measures are employed to analyze the data distribution and summarization style shift characteristics.
- Seven popular PLMs and LLMs are evaluated on the benchmark in both in-domain and cross-domain settings.
- Models are evaluated across different domain shifts for future research references.

Please check more details in our paper [DomainSum: A Hierarchical Benchmark for Fine-Grained Domain Shift in Abstractive Text Summarization](https://arxiv.org/abs/2410.15687).

## Data

The dataset (including the sampled dataset) can be obtained from [Google Drive](https://drive.google.com/drive/folders/1rNp8PZg9iADISCjApvP4LY9oCbR0wbi0?usp=sharing).

## Preparation for running
1. Create the conda environment
    ```
    conda create -n domainsum python=3.10
    ```
2. Activate the conda environment
    ```
    conda activate domainsum
    ```
3. Install pytorch. Please check your CUDA version before the installation and modify it accordingly, or you can refer to [pytorch website](https://pytorch.org)
    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    ```
3. Install requirements
    ```
    pip install -r requirements.txt

## Running

### Data Statistics Collection

```
python ./get_statistics.py
```

To analyze dataset characteristics, use `get_statistics.py` that extracts the 8 key measures used in our paper.


### Test Zero-shot/Two-shot Summarization with LLMs (except GPT-4)

```
# Testing 0-shot text summarization
./zero-shot/run_0shot.sh [config]

# Testing 2-shot text summarization
./two-shot/run_2shot.sh [config]
```

The output will be a JSON file with prediction results, which can be evaluated using our `json_evaluation.py` script.

### Test Zero-shot/Two-shot Summarization with GPT-4

```
# Testing 0-shot text summarization
./GPT/GPT_api.sh [config]

# Testing 2-shot text summarization
./GPT/GPT_api_2shot.sh [config]
```

The output will also be a JSON file with prediction results, which can be evaluated using our `json_evaluation.py` script.

### Fine-tuning Pretrained Language Models (PLMs)

```
# Fine-tuning BART
./finetune/run_bart.sh [config]

# Fine-tuning PEGASUS-X
./finetune/run_pegasusx.sh [config]

# Testing Cross-domain performance with PLMs
./finetune/do_predict.sh [config]
```

Fine-tuning scripts automatically save model checkpoints. To evaluate cross-domain performance, you can use a model checkpoint from the source domain to test on a target domain dataset.

### Fine-tuning Llama3

```
# Pre-processing dataset for fine-tuning Llama
python ./finetune-Llama/data/data_process4llama.py

# Fine-tuning Llama3
./finetune-Llama/train.sh [config]

# Generating predictions with fine-tuned Llama3
./finetune-Llama/predic.sh [config]
```

#### Accessing Llama3

We use Hugging Face to access the Llama3 model and LoRA for fine-tuning. To gain access, complete the official authorization process, then enter your authorized Hugging Face API key, associated with your permitted account, into the `sh` files to successfully run the fine-tuning scripts. For further details, please refer to the official Meta documentation: [Fine-Tuning Guide](https://www.llama.com/docs/how-to-guides/fine-tuning/) and the Hugging Face documentation: [Llama3.1-8B Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

### Get Evaluation Metrics

```
python json_evaluation.py
```

Before running the evaluation, insert the path of your JSON files as the `json_files` argument in the `json_evaluation.py` script.

## Citation

If these data and codes help you, please cite our paper.

```bib
@misc{yuan2024domainsumhierarchicalbenchmarkfinegrained,
      title={DomainSum: A Hierarchical Benchmark for Fine-Grained Domain Shift in Abstractive Text Summarization}, 
      author={Haohan Yuan and Haopeng Zhang},
      year={2024},
      eprint={2410.15687},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.15687}, 
}
```
