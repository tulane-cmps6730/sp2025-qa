from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Replace MODEL_NAME with one of: "BERTcheckpoint-84000", "DistilBERTcheckpoint-69000", 
# "Albertcheckpoint-37500", or "RoBERTacheckpoint-33000"
model_path = f"FinalModels/DistilBERTcheckpoint-69000/"

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