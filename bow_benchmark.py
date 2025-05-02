import collections
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import evaluate as eval_lib
from datasets import load_dataset
from tqdm import tqdm

# Load evaluation metric
squad_metric = eval_lib.load("squad_v2")  # For compatibility with both datasets

class BagOfWordsBenchmark:
    def __init__(self, max_answer_length=30, window_size=50, answerable_threshold=0.15):
        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        self.max_answer_length = max_answer_length
        self.window_size = window_size
        self.answerable_threshold = answerable_threshold  # Threshold to determine if a question is answerable
        
    def find_answer_with_confidence(self, question, context):
        """
        Find the most likely answer span using Bag of Words approach.
        Also returns a confidence score to determine if the question is answerable.
        """
        # Handle empty contexts
        if not context or not question:
            return "", 0.0
        
        # Split context into sentences for more efficient processing
        sentences = context.replace(".", ". ").replace("?", "? ").replace("!", "! ").split(". ")
        
        # Find the most similar sentence to the question
        best_score = -1
        best_sent = ""
        
        try:
            # Vectorize the question once
            q_vec = self.vectorizer.fit_transform([question])
            
            # Find the most similar sentence
            if sentences:
                sent_vec = self.vectorizer.transform(sentences)
                similarities = cosine_similarity(q_vec, sent_vec)[0]
                best_idx = np.argmax(similarities)
                best_sent = sentences[best_idx]
                best_score = similarities[best_idx]
        except ValueError as e:
            # Handle case where question contains only stop words
            if "empty vocabulary" in str(e):
                # If question has only stop words, return the first sentence as a fallback with low confidence
                return sentences[0].strip() if sentences else "", 0.05
            else:
                raise
        
        # If no good match found, return empty string with low confidence
        if best_score < 0.1:
            return "", 0.0
            
        # Split the best sentence into words
        words = best_sent.split()
        
        # If sentence is short, return it directly
        if len(words) <= self.max_answer_length:
            return best_sent.strip(), best_score
        
        # For longer sentences, find the best window within it
        best_window_score = -1
        best_window = ""
        
        # Use sliding window instead of checking all possible spans
        for i in range(len(words) - min(self.max_answer_length, 5) + 1):
            # Only check windows of several reasonable sizes, not all possible sizes
            for window_size in [3, 5, 10, 15, min(self.max_answer_length, len(words) - i)]:
                if i + window_size > len(words):
                    continue
                    
                window = ' '.join(words[i:i + window_size])
                
                # Skip windows that are too short
                if len(window) < 2:
                    continue
                    
                # Compute similarity of this window to the question
                try:
                    window_vec = self.vectorizer.transform([window])
                    similarity = cosine_similarity(q_vec, window_vec)[0][0]
                    
                    if similarity > best_window_score:
                        best_window_score = similarity
                        best_window = window
                except:
                    continue
        
        # Return the best window if found, otherwise the best sentence, along with confidence
        if best_window and best_window_score > 0.1:
            return best_window, best_window_score
        
        return best_sent.strip(), best_score
    
    def predict(self, examples):
        """Make predictions for the entire dataset using batched processing"""
        predictions = []
        
        for ex in tqdm(examples, desc="Predicting"):
            # Get answer and confidence score
            pred_text, confidence = self.find_answer_with_confidence(ex["question"], ex["context"])
            
            # Determine if the question is answerable based on confidence
            # If confidence is below threshold, predict no answer
            no_answer_probability = 1.0 - min(confidence / 0.5, 1.0)  # Scale confidence to [0,1]
            
            # If confidence is very low, predict no answer
            if confidence < self.answerable_threshold:
                pred_text = ""
                no_answer_probability = 1.0
            
            predictions.append({
                "id": ex["id"],
                "prediction_text": pred_text,
                "no_answer_probability": no_answer_probability
            })
            
        return predictions
    
    def evaluate(self, dataset_name, split="validation", subset=None, sample_size=None):
        """Evaluate the model on a dataset and return metrics"""
        print(f"\nEvaluating on {dataset_name}{f' ({subset})' if subset else ''} {split} set...")
        
        # Load dataset
        if dataset_name == "squad_adversarial":
            examples = load_dataset("stanfordnlp/squad_adversarial", subset, trust_remote_code=True)[split]
        else:
            examples = load_dataset(dataset_name)[split]
        
        # Optionally use a smaller sample for faster debugging
        if sample_size and sample_size < len(examples):
            random.seed(42)  # For reproducibility
            indices = random.sample(range(len(examples)), sample_size)
            examples = examples.select(indices)
            print(f"Using {sample_size} examples for faster evaluation")
        
        # Make predictions
        predictions = self.predict(examples)
        
        # Format references
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        
        # Calculate metrics
        results = squad_metric.compute(predictions=predictions, references=references)
        em_key = "exact_match" if "exact_match" in results else "exact"
        print(f"Exact Match: {results[em_key]:.2f}")
        print(f"F1 Score: {results['f1']:.2f}")
        
        # Get examples of wrong predictions
        wrong = []
        for ex, pred in zip(examples, predictions):
            is_answerable = len(ex["answers"]["text"]) > 0
            is_correct = False
            
            if is_answerable:
                # For answerable questions, check if prediction matches any answer
                is_correct = pred["prediction_text"] in ex["answers"]["text"]
            else:
                # For unanswerable questions, check if prediction is empty
                is_correct = pred["prediction_text"] == ""
                
            if not is_correct:
                wrong.append({
                    "question": ex["question"],
                    "pred": pred["prediction_text"],
                    "gold": ex["answers"]["text"][0] if ex["answers"]["text"] else "No answer",
                    "context": ex["context"],
                    "answerable": is_answerable,
                    "confidence": 1.0 - pred["no_answer_probability"]
                })
        
        print("\n3 Examples of Wrong Predictions:")
        if wrong:
            for i, s in enumerate(random.sample(wrong, min(3, len(wrong))), 1):
                print(f"\nExample {i}:")
                print(f"Question: {s['question']}")
                print(f"Predicted: '{s['pred']}' (Confidence: {s['confidence']:.2f})")
                print(f"Expected: '{s['gold']}'")
                print(f"Answerable: {s['answerable']}")
        else:
            print("No wrong predictions found or no answerable questions in the sample!")
        
        return results
    
def main():
    # Initialize the model
    model = BagOfWordsBenchmark(max_answer_length=30, answerable_threshold=0.15)
    
    # Evaluate on SQuAD 1.1
    squad1_results = model.evaluate("squad")
    
    # Evaluate on SQuAD 2.0
    squad2_results = model.evaluate("squad_v2")
    
    # Evaluate on AddSent
    addsent_results = model.evaluate("squad_adversarial", subset="AddSent")
    
    # Evaluate on AddOneSent
    addonesent_results = model.evaluate("squad_adversarial", subset="AddOneSent")
    
    # Print summary
    print("\n========== SUMMARY ==========")
    print("Bag of Words Benchmark Model Results:")
    print(f"SQuAD 1.1: EM={squad1_results['exact_match' if 'exact_match' in squad1_results else 'exact']:.2f}, F1={squad1_results['f1']:.2f}")
    print(f"SQuAD 2.0: EM={squad2_results['exact_match' if 'exact_match' in squad2_results else 'exact']:.2f}, F1={squad2_results['f1']:.2f}")
    print(f"AddSent: EM={addsent_results['exact_match' if 'exact_match' in addsent_results else 'exact']:.2f}, F1={addsent_results['f1']:.2f}")
    print(f"AddOneSent: EM={addonesent_results['exact_match' if 'exact_match' in addonesent_results else 'exact']:.2f}, F1={addonesent_results['f1']:.2f}")

if __name__ == "__main__":
    main() 