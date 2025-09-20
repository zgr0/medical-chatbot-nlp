import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import evaluate
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from datasets import Dataset
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def load_model_and_tokenizer(model_path="./t5_chatbot_model", tokenizer_path="./t5_chatbot_tokenizer"):
    """
    Load the trained model and tokenizer
    """
    print("[INFO] Loading model and tokenizer...")
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()  # Set model to evaluation mode
        
        print(f"[INFO] Model loaded successfully and moved to {device}")
        return model, tokenizer, device
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        return None, None, None

def generate_answer(question, model, tokenizer, device, max_length=64):
    """
    Generate an answer for a given question
    """
    # Prepare input
    input_text = f"answer the following question: {question}"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).input_ids.to(device)
    
    # Generate answer
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    # Decode and return answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def evaluate_model(model, tokenizer, device, test_data_path="./HC_DATA/medquad_qa_pairs.csv", num_samples=None):
    """
    Evaluate model performance on test dataset
    """
    print("[INFO] Loading test data...")
    df = pd.read_csv(test_data_path)
    if num_samples:
        df = df.sample(n=num_samples, random_state=42)
    
    # Initialize metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    
    predictions = []
    references = []
    exact_matches = 0
    
    print("[INFO] Generating predictions...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row['question']
        true_answer = row['answer']
        
        # Generate prediction
        pred_answer = generate_answer(question, model, tokenizer, device)
        
        predictions.append(pred_answer)
        references.append(true_answer)
        
        # Check for exact match
        if pred_answer.strip().lower() == true_answer.strip().lower():
            exact_matches += 1
    
    # Calculate metrics
    exact_match_score = exact_matches / len(df)
    bleu_score = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
    rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Number of test samples: {len(df)}")
    print(f"Exact Match Score: {exact_match_score:.4f}")
    print(f"BLEU Score: {bleu_score['bleu']:.4f}")
    print(f"ROUGE-L Score: {rouge_scores['rougeL']:.4f}")
    
    return {
        'exact_match': exact_match_score,
        'bleu': bleu_score['bleu'],
        'rougeL': rouge_scores['rougeL']
    }

def interactive_mode(model, tokenizer, device):
    """
    Interactive mode for testing the model with user input
    """
    print("\n=== Interactive Mode ===")
    print("Enter your medical questions (type 'quit' to exit)")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() == 'quit':
            break
            
        if not question:
            print("Please enter a valid question!")
            continue
            
        print("\nGenerating answer...")
        answer = generate_answer(question, model, tokenizer, device)
        print(f"\nAnswer: {answer}")

def main():
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer()
    if model is None:
        return
    
    while True:
        print("\n=== Medical Chatbot Evaluation ===")
        print("1. Run model evaluation on test set")
        print("2. Interactive mode")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            num_samples = input("Enter number of test samples (press Enter for all): ").strip()
            num_samples = int(num_samples) if num_samples.isdigit() else None
            evaluate_model(model, tokenizer, device, num_samples=num_samples)
        
        elif choice == '2':
            interactive_mode(model, tokenizer, device)
        
        elif choice == '3':
            break
        
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()