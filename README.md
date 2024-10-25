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

## Running

### Test with LLMs (except GPT-4o)

```
# Testing 0-shot text summarization
python scripts/0shot/run_0shot.py --model [model_name] --input [test_data] --output [result_saving_folder] --delay 0 --token [your hugging face API token]

# Testing 2-shot text summarization
python scripts/2shot/run_2shot.py --model [model_name] --input [test_data] --extra [example_data] --output [result_saving_folder] --delay 0 --token [your hugging face API token]
```
Quick start examples can be found in ``run0shot_command.txt`` and ``run2shot_command.txt``. The output will be a JSON file with prediction results, which can be evaluated using our evaluation script.

### Get Evaluation Metrics
```
python scripts/json_evaluation.py
```
Before evaluation, you need to modify the ``json_files`` argument and insert the path to your prediction result JSON file in the ``json_evaluation.py`` script.

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
