# Question Answering with Transformer Models

> **Note**: The complete code for this project is available in the [sp2025-qa-code repository](https://github.com/tulane-cmps6730/sp2025-qa-code).

## Project Overview
This project explores the effectiveness of various transformer-based models for question answering on the SQuAD dataset. We fine-tuned multiple pre-trained models, evaluated their performance, and analyzed their behavior under adversarial conditions.

![BERT Training Progress](FurtherTrainingPlots/BERT_training_progress.png)

## Models Explored
The project evaluated four different transformer architectures:

- **BERT**: Bidirectional Encoder Representations from Transformers
- **DistilBERT**: A distilled version of BERT with ~40% fewer parameters
- **Albert**: A Lite BERT with significantly fewer parameters
- **RoBERTa**: A robustly optimized BERT approach

## Key Findings
- BERT achieved the highest F1 score (~73.7%) after 84,000 training steps
- Smaller models like DistilBERT and Albert offered competitive performance with reduced computational requirements
- All models showed vulnerability to adversarial examples, particularly with question modification
- Model performance varied across different question types and context lengths

![Albert Training Progress](FurtherTrainingPlots/albert_training_progress.png)

## Evaluation Methods
We evaluated models using:
- Exact Match and F1 metrics for answer accuracy
- Performance across different question types
- Resilience to adversarial examples
- Inference speed and model size considerations

![DistilBERT Training Progress](FurtherTrainingPlots/DistilbertTraining.png)

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

This project demonstrates the effectiveness of transformer-based models for question answering tasks. While BERT-based models achieve strong performance, smaller models like DistilBERT and Albert offer compelling alternatives when computational resources are limited. Future work could explore hybrid approaches and additional techniques to improve resilience against adversarial attacks. 