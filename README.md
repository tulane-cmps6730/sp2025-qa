# Question Answering with Transformer Models

## Project Overview
This project explores the effectiveness of various transformer-based models for question answering on the SQuAD dataset. We fine-tuned multiple pre-trained models, evaluated their performance, and analyzed their behavior under standard and adversarial conditions.

## Goals
- Evaluate the performance of different transformer architectures on question answering tasks
- Compare model performance across both SQuAD 1.1 and SQuAD 2.0 datasets
- Assess model robustness against adversarial examples
- Identify optimal trade-offs between model size, computational requirements, and performance

## Methods
- Fine-tuned four state-of-the-art transformer models (BERT, DistilBERT, ALBERT, and RoBERTa)
- Evaluated on standard benchmarks (SQuAD 1.1, SQuAD 2.0)
- Tested with adversarial attacks (AddSent and AddOneSent)
- Analyzed performance across different question types and answer contexts

## Models Explored
The project evaluated four different transformer architectures:

- **BERT**: Bidirectional Encoder Representations from Transformers
- **DistilBERT**: A distilled version of BERT with ~40% fewer parameters
- **ALBERT**: A Lite BERT with significantly fewer parameters
- **RoBERTa**: A robustly optimized BERT approach

## Model Performance

Performance metrics across different evaluation settings (EM/F1 scores):

| Model | SQuAD 1.1 (EM/F1) | SQuAD 2.0 (EM/F1) | AddSent (EM/F1) | AddOneSent (EM/F1) |
|-------|------------------|-------------------|----------------|-------------------|
| ALBERT | 76.39 / 83.22 | 75.67 / 79.26 | 55.84 / 60.52 | 64.13 / 69.70 |
| DistilBERT | 60.70 / 67.99 | 66.27 / 70.08 | 40.87 / 46.05 | 47.68 / 53.89 |
| RoBERTa | 71.16 / 82.93 | 75.67 / 81.58 | 50.90 / 59.74 | 59.15 / 69.31 |
| BERT | 70.58 / 77.71 | 71.72 / 75.53 | 50.65 / 56.22 | 57.97 / 64.02 |

![BERT Training Progress](FurtherTrainingPlots/BERT_training_progress.png)
![Albert Training Progress](FurtherTrainingPlots/albert_training_progress.png)
![DistilBERT Training Progress](FurtherTrainingPlots/DistilbertTraining.png)

## Evaluation Metrics

We evaluated our models using two primary metrics:

- **Exact Match (EM)**: The percentage of predictions that exactly match any of the ground truth answers. This is a strict binary measure where a prediction is either correct or incorrect.

- **F1 Score (F1)**: The harmonic mean of precision and recall, treating the prediction and ground truth answers as bags of tokens. This metric provides a more flexible measure that rewards partial matches, which is especially important for longer answers.

Both metrics are reported as percentages, with higher values indicating better performance.

## Key Findings
- ALBERT achieved the highest overall performance on SQuAD 1.1 (83.22% F1) and remained strong on SQuAD 2.0
- RoBERTa demonstrated exceptional robustness on SQuAD 2.0 (81.58% F1)
- All models showed vulnerability to adversarial examples, with performance dropping significantly on the AddSent challenge
- ALBERT maintained the best performance under adversarial conditions, highlighting its superior generalization capabilities

## Project Structure

The repository is organized as follows:

- **Initial_Train/**: Python scripts for initial model training on SQuAD datasets
  - Contains training scripts for each model architecture (BERT, DistilBERT, ALBERT, RoBERTa)

- **FurtherTrainingipynb/**: Jupyter notebooks for continued training and optimization
  - Fine-tuning notebooks for each model with advanced parameter settings

- **FurtherTrainingPlots/**: Visualizations of training progress
  - Learning curves and performance metrics during extended training

- **FurtherTrainingMetricsJsons/**: JSON files containing detailed training metrics
  - Raw metrics data for analysis and comparison

- **Eval_1.1_Ipynb/**: Notebooks for evaluating models on SQuAD 1.1
  - Model-specific evaluation scripts and results analysis

- **Eval_2.0.ipynb/**: Notebooks for evaluating models on SQuAD 2.0
  - Tests focused on handling unanswerable questions

- **Eval_Adversarial.ipynb/**: Notebooks for adversarial evaluation
  - Tests with AddSent and AddOneSent attack strategies
  - To switch between adversarial datasets, modify the dataset loading line:
    ```python
    examples = load_dataset("stanfordnlp/squad_adversarial", "AddSent", trust_remote_code=True)["validation"]
    ```
  - Simply change "AddSent" to "AddOneSent" to evaluate on the alternative adversarial dataset

- **FinalModels/**: Trained model checkpoints (available through GitHub Releases)
  - Optimized model weights for each architecture

## Adversarial Evaluation

The project includes comprehensive evaluation against two types of adversarial attacks:

1. **AddSent**: Adds a distracting sentence to the context that looks similar to the question but contains a different answer.
2. **AddOneSent**: Adds a single adversarial sentence that doesn't answer the question but could mislead the model.

As shown in the performance table, all models experience performance degradation under adversarial conditions, with the AddSent attack being particularly challenging. ALBERT demonstrates the most resilience against these attacks, maintaining the highest F1 scores in both adversarial scenarios.

To run your own adversarial evaluations, use the notebooks in the Eval_Adversarial.ipynb directory and switch between datasets by modifying the dataset loading parameter.

## Model Downloads

Due to GitHub file size limitations, the trained models are available through the GitHub Release section, split into two parts:

**[Download Models from Releases](https://github.com/tulane-cmps6730/sp2025-qa/releases)**

Part 1 (1.2GB):
- BERT checkpoint (84000 steps)
- Albert checkpoint (37500 steps)

Part 2 (1.9GB):
- RoBERTa checkpoint (33000 steps)
- DistilBERT checkpoint (69000 steps)

### Installation Instructions

1. Download both zip files from the Releases page
2. Extract them to your project directory:
   ```bash
   unzip FinalModels_Part1.zip
   unzip FinalModels_Part2.zip
   ```
3. The files will automatically merge into the correct `FinalModels` directory structure

## Usage

To use the models for inference:

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Replace MODEL_NAME with one of: "BERTcheckpoint-84000", "DistilBERTcheckpoint-69000", 
# "Albertcheckpoint-37500", or "RoBERTacheckpoint-33000"
model_path = f"FinalModels/{MODEL_NAME}/"

# Load model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Example question and context
question = "What is the capital of France?"
context = "Paris is the capital and most populous city of France."

# Tokenize input
inputs = tokenizer(question, context, return_tensors="pt")

# Get model prediction
outputs = model(**inputs)
answer_start = outputs.start_logits.argmax()
answer_end = outputs.end_logits.argmax() + 1
answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])

print(f"Answer: {answer}")
```

## Requirements

```
torch>=2.1          
transformers>=4.39
datasets>=2.18
evaluate>=0.4
accelerate>=0.27
tqdm
```

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Conclusion

Our comprehensive evaluation demonstrates that transformer-based models achieve impressive performance on question answering tasks, with ALBERT showing particularly strong results across all test conditions. While all models exhibit vulnerability to adversarial examples, the relative performance maintained by ALBERT suggests promising directions for improving model robustness.

The trade-offs between model size and performance are evident, with lighter models like DistilBERT offering reasonable performance with significantly reduced computational requirements. This suggests that for many practical applications, smaller models may provide an optimal balance of accuracy and efficiency.

Future work could explore hybrid approaches and additional techniques to improve resilience against adversarial attacks. 
