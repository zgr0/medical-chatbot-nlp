import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import evaluate
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from datasets import Dataset
import warnings
import os
import time
from torch.cuda.amp import autocast  # For mixed precision
import concurrent.futures
from functools import partial

# Suppress warnings
warnings.filterwarnings("ignore")

def load_model_and_tokenizer(model_path="./t5_chatbot_model/model", tokenizer_path=None):
    """
    Load the trained model and tokenizer with optimized settings
    """
    print("[INFO] Loading model and tokenizer...")
    start_time = time.time()

    try:
        # Use model path for tokenizer if not specified
        if tokenizer_path is None:
            # Default to tokenizer directory if it exists, otherwise use model path
            if os.path.exists("./t5_chatbot_model/tokenizer"):
                tokenizer_path = "./t5_chatbot_model/tokenizer"
            else:
                tokenizer_path = model_path

        # Check if we're using the peft adapter, which requires base model
        using_peft = "peft_adapter" in model_path
        if using_peft:
            print("[INFO] Loading PEFT adapter with base model...")
            base_model_path = "./t5_chatbot_model/model"

        # Configure GPU settings for evaluation with stability focus
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False  # Disable for more stable memory usage
            torch.backends.cudnn.allow_tf32 = False  # Disable to avoid precision-related errors
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.cuda.empty_cache()
            
            # Check available GPU memory
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            print(f"[INFO] Available GPU memory: {free_memory / 1024**3:.2f} GB")
            
            # Force garbage collection
            import gc
            gc.collect()

        # Check if model path exists
        if not os.path.exists(model_path):
            print(f"[ERROR] Model path does not exist: {model_path}")
            return None, None, None

        # Log which files are present in the directory
        model_files = os.listdir(model_path)
        print(f"[INFO] Found {len(model_files)} files in model directory: {', '.join(model_files)}")

        # Check if safetensors file is present
        has_safetensors = any(f.endswith('.safetensors') for f in model_files)
        print(f"[INFO] Using {'safetensors' if has_safetensors else 'pytorch'} model format")

        # Load model with low CPU memory usage and move to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check if we're using the peft adapter
        if using_peft:
            try:
                # Try to import PEFT library
                try:
                    from peft import PeftModel, PeftConfig
                except ImportError:
                    print("[WARNING] PEFT library not found. Installing...")
                    import pip
                    pip.main(['install', 'peft'])
                    from peft import PeftModel, PeftConfig
                
                # Load base model first
                base_model_path = "./t5_chatbot_model/model"
                print(f"[INFO] Loading base model from {base_model_path}")
                model = T5ForConditionalGeneration.from_pretrained(
                    base_model_path,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
                
                # Then load PEFT adapter
                print(f"[INFO] Loading PEFT adapter from {model_path}")
                model = PeftModel.from_pretrained(model, model_path)
                print("[INFO] Successfully loaded PEFT model")
            except Exception as e:
                print(f"[ERROR] Failed to load PEFT model: {str(e)}")
                print("[INFO] Falling back to regular model loading...")
                model = T5ForConditionalGeneration.from_pretrained(
                    model_path, 
                    device_map="auto" if torch.cuda.is_available() else None,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
        else:
            # Load regular model
            model = T5ForConditionalGeneration.from_pretrained(
                model_path, 
                device_map=None,  # Don't use auto device mapping to prevent memory fragmentation
                torch_dtype=torch.float32,  # Use FP32 to avoid CUBLAS issues with FP16
                low_cpu_mem_usage=True
            )
        model.eval()  # Set model to evaluation mode
        
        # Optimize memory usage
        if device.type == "cuda":
            model = model.to(device)
        
        # Load tokenizer with caching
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, use_fast=True)
        
        load_time = time.time() - start_time
        print(f"[INFO] Model loaded successfully in {load_time:.2f} seconds")
        print(f"[INFO] Using device: {device} | Mixed Precision: {device.type == 'cuda'}")
        return model, tokenizer, device
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        return None, None, None

def generate_answer(question, model, tokenizer, device, max_length=256):
    """
    Generate an answer for a given question with memory-efficient settings
    """
    # Format input to be more specific and encourage detailed responses
    input_text = f"Medical question: {question.strip()}\nProvide a comprehensive, detailed medical answer:"
    
    # Print first question and input text for debugging
    if question == "What is (are) Glaucoma ?":
        print(f"[DEBUG] Sample question: '{question}'")
        print(f"[DEBUG] Input format: '{input_text}'")
    
    encoding = tokenizer(input_text, return_tensors="pt", max_length=256, 
                         truncation=True, padding="max_length")
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)
    
    # Generate answer with memory-efficient settings
    with torch.no_grad():
        if device.type == "cuda":
            # Avoid mixed precision which can cause CUBLAS errors
            torch.cuda.empty_cache()  # Clear cache before generation
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=384,  # Reduced max length to save memory
                min_length=50,   # Reduced min length requirement
                num_beams=2,     # Reduced beam search to save memory
                length_penalty=1.5,  # Still favor longer outputs but less aggressively
                    early_stopping=False,  # Let it generate to full potential
                    no_repeat_ngram_size=3,  # Increased for better fluency
                    do_sample=False,  # Deterministic generation
                    temperature=1.0,  # Default temperature
                    top_k=50,  # Filter to top 50 tokens
                    repetition_penalty=1.5,  # Strongly discourage repetition
                    use_cache=True,  # Ensure caching is enabled
                    return_dict_in_generate=False
                )
        else:
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=512,  # Significantly increased max length
                min_length=100,  # Force longer outputs
                num_beams=5,  # Increased beam search for better quality
                length_penalty=2.5,  # Strongly favor longer outputs
                early_stopping=False,  # Let it generate to full potential
                no_repeat_ngram_size=3,  # Increased for better fluency
                do_sample=False,  # Deterministic generation 
                temperature=1.0,  # Default temperature
                top_k=50,  # Filter to top 50 tokens
                repetition_penalty=1.5,  # Strongly discourage repetition
                use_cache=True
            )
    
    # Decode and return answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def generate_batch_answers(batch_questions, model, tokenizer, device, max_length=64, batch_size=2):
    """
    Generate answers for a batch of questions with memory-efficient settings
    """
    # Use medical domain specific prompt with emphasis on detailed responses
    batch_inputs = [f"Medical question: {q.strip()}\nProvide a comprehensive, detailed medical answer:" for q in batch_questions]
    
    # Tokenize all questions at once
    encodings = tokenizer(
        batch_inputs, 
        return_tensors="pt", 
        max_length=256,  # Reduced to save memory
        truncation=True, 
        padding="max_length"
    )
    
    # Process in smaller batches to avoid OOM - reduced batch size
    all_outputs = []
    for i in range(0, len(batch_questions), batch_size):
        batch_input_ids = encodings.input_ids[i:i+batch_size].to(device)
        batch_attention_mask = encodings.attention_mask[i:i+batch_size].to(device)
        
        # Clear cache between batches
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
        # Generate answers with memory-efficient settings
        with torch.no_grad():
                if device.type == "cuda":
                    # Don't use autocast to avoid CUBLAS errors
                    outputs = model.generate(
                        batch_input_ids,
                        attention_mask=batch_attention_mask,
                        max_length=384,  # Reduced for memory efficiency
                        num_beams=2,     # Reduced beam search
                        length_penalty=1.5,  # Still favor longer answers but less aggressively
                        early_stopping=True,  # Enable early stopping to save memory
                        no_repeat_ngram_size=3,
                        use_cache=True,
                        do_sample=False,  # Deterministic generation
                        temperature=1.0,   # No temperature adjustment
                        min_length=50,     # Reduced minimum length
                        repetition_penalty=1.2  # Reduced penalty for memory efficiency
                    )
                else:
                    outputs = model.generate(
                        batch_input_ids,
                        attention_mask=batch_attention_mask,
                        max_length=512,  # Greatly increased to match CUDA path
                        num_beams=5,  # Match CUDA path
                        length_penalty=2.5,  # Strong penalty to favor longer answers
                        early_stopping=False,  # Let generation reach full potential
                        no_repeat_ngram_size=3,
                        use_cache=True,
                        do_sample=False,  # Deterministic generation
                        temperature=1.0,   # No temperature adjustment
                        min_length=100,  # Enforce substantial answers
                        repetition_penalty=1.5  # Strongly penalize repetition
                    )
                
        all_outputs.extend(outputs)
    
    # Decode all outputs at once
    batch_answers = tokenizer.batch_decode(all_outputs, skip_special_tokens=True)
    return batch_answers


def process_subset(subset_df, model, tokenizer, device):
    """
    Process a subset of data for parallel execution with memory optimizations
    """
    subset_predictions = []
    subset_references = []
    subset_exact_matches = 0
    
    questions = subset_df['question'].tolist()
    answers = subset_df['answer'].tolist()
    
    # Use batch processing with much smaller batch size
    batch_size = 1  # Process one question at a time to minimize memory usage
    
    try:
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_true_answers = answers[i:i+batch_size]
            
            # Clear GPU cache before each batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
            # Generate answers with optimized function
            batch_pred_answers = generate_batch_answers(batch_questions, model, tokenizer, device)
            
            subset_predictions.extend(batch_pred_answers)
            subset_references.extend(batch_true_answers)
            
            # Check for exact matches
            for pred, true in zip(batch_pred_answers, batch_true_answers):
                if pred.strip().lower() == true.strip().lower():
                    subset_exact_matches += 1
                    
            # Print progress every few batches
            if i % 5 == 0:
                print(f"[INFO] Processed {i}/{len(questions)} questions")
    except Exception as e:
        print(f"[ERROR] Error during processing: {e}")
        print("[INFO] Returning partial results...")
    
    return subset_predictions, subset_references, subset_exact_matches
    
    return subset_predictions, subset_references, subset_exact_matches


def evaluate_model(model, tokenizer, device, test_data_path="./HC_DATA/medquad_qa_pairs.csv", num_samples=None):
    """
    Evaluate model performance on test dataset with enhanced debugging and metrics
    """
    start_time = time.time()
    print("\n=== Evaluation Results ===")
    print("[INFO] Loading test data...")
    df = pd.read_csv(test_data_path)
    
    # Display data format for debugging
    print("\n[DEBUG] Dataset columns:", df.columns.tolist())
    
    # Show sample data in a more readable format
    print("\n[DEBUG] Sample questions from test set:")
    for i, (q, a) in enumerate(zip(df['question'].iloc[:3], df['answer'].iloc[:3])):
        q_display = q[:100] + ('...' if len(q) > 100 else '')
        a_display = a[:100] + ('...' if len(a) > 100 else '')
        print(f"Q{i+1}: {q_display}")
        print(f"A{i+1}: {a_display}\n")
    
    # Always limit sample size to avoid CUDA errors
    if num_samples:
        max_samples = min(num_samples, len(df))
    else:
        # Default to a conservative number for GPU memory constraints
        max_samples = min(20, len(df))
    
    df = df.sample(n=max_samples, random_state=42)
    
    print(f"[INFO] Evaluating model on {len(df)} samples using {device.type.upper()}...")
    
    # Determine if we should use parallel processing
    use_parallel = len(df) > 100 and device.type == 'cpu'
    num_workers = os.cpu_count() if use_parallel else 1
    
    predictions = []
    references = []
    exact_matches = 0
    
    if use_parallel and num_workers > 1:
        # Split data for parallel processing
        print(f"[INFO] Using parallel processing with {num_workers} workers...")
        split_dfs = np.array_split(df, num_workers)
        
        process_fn = partial(process_subset, model=model, tokenizer=tokenizer, device=device)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(process_fn, split_dfs), total=len(split_dfs), desc="Processing batches"))
            
        for batch_preds, batch_refs, batch_exact_matches in results:
            predictions.extend(batch_preds)
            references.extend(batch_refs)
            exact_matches += batch_exact_matches
    else:
        # Use batch processing but single process
        print("[INFO] Using optimized batch processing...")
        batch_size = 16 if device.type == "cuda" else 8
        questions = df['question'].tolist()
        true_answers = df['answer'].tolist()
        
        for i in tqdm(range(0, len(df), batch_size), desc="Generating predictions"):
            batch_questions = questions[i:i+batch_size]
            batch_true_answers = true_answers[i:i+batch_size]
            
            batch_pred_answers = generate_batch_answers(
                batch_questions, model, tokenizer, device, batch_size=batch_size
            )
            
            predictions.extend(batch_pred_answers)
            references.extend(batch_true_answers)
            
            # Check for exact matches
            for pred, true in zip(batch_pred_answers, batch_true_answers):
                if pred.strip().lower() == true.strip().lower():
                    exact_matches += 1
    
    # Initialize metrics
    print("[INFO] Computing evaluation metrics...")
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    
    # Calculate metrics
    exact_match_score = exact_matches / len(df)
    bleu_score = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
    rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
    
    # Calculate time taken
    eval_time = time.time() - start_time
    samples_per_second = len(df) / eval_time
    
    # Print results and sample predictions
    print("\n=== Evaluation Results ===")
    print(f"Number of test samples: {len(df)}")
    print(f"Evaluation time: {eval_time:.2f} seconds ({samples_per_second:.2f} samples/sec)")
    print(f"Exact Match Score: {exact_match_score:.4f}")
    print(f"BLEU Score: {bleu_score['bleu']:.4f}")
    print(f"ROUGE-L Score: {rouge_scores['rougeL']:.4f}")
    
    # Show detailed sample predictions for better debugging
    print("\n=== Sample Predictions ===")
    for i in range(min(5, len(predictions))):
        print(f"\nQuestion: {df['question'].iloc[i]}")
        
        # Calculate length information
        pred_length = len(predictions[i].split())
        ref_length = len(references[i].split())
        length_diff = pred_length - ref_length
        
        print(f"Predicted ({pred_length} words): {predictions[i]}")
        print(f"Reference ({ref_length} words): {references[i]}")
        print(f"Length difference: {length_diff} words ({length_diff/max(1, ref_length):.1%} of reference)")
        
        # Check for substring matches to see if there's any partial success
        pred_lower = predictions[i].lower()
        ref_lower = references[i].lower()
        
        # Calculate word overlap
        pred_words = set(pred_lower.split())
        ref_words = set(ref_lower.split())
        common_words = pred_words.intersection(ref_words)
        overlap_ratio = len(common_words) / len(ref_words) if ref_words else 0
        
        print(f"Word overlap: {len(common_words)}/{len(ref_words)} words ({overlap_ratio:.1%})")
        print(f"Exact Match: {predictions[i].strip().lower() == references[i].strip().lower()}")
    
    # Show common tokens in predictions with percentages
    all_preds_text = " ".join(predictions)
    word_counts = {}
    for word in all_preds_text.lower().split():
        if len(word) > 3:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Get top repeated words
    common_words = sorted([(word, count) for word, count in word_counts.items() 
                          if count > len(df)/10], 
                         key=lambda x: x[1], reverse=True)[:15]
    
    if common_words:
        print("\n=== Common Words in Predictions ===")
        for word, count in common_words:
            percentage = count / len(predictions) * 100
            print(f"'{word}': {count} times ({percentage:.1f}% of predictions)")
    else:
        print("\nNo common repeated words detected")
    
    return {
        'exact_match': exact_match_score,
        'bleu': bleu_score['bleu'],
        'rougeL': rouge_scores['rougeL'],
        'eval_time': eval_time,
        'samples_per_second': samples_per_second
    }

def interactive_mode(model, tokenizer, device):
    """
    Interactive mode for testing the model with user input
    """
    print("\n=== Interactive Mode ===")
    print("Enter your medical questions (type 'quit' to exit)")
    
    # Warm up the model to make subsequent generations faster
    print("[INFO] Warming up the model...")
    _ = generate_answer("What is diabetes?", model, tokenizer, device)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() == 'quit':
            break
            
        if not question:
            print("Please enter a valid question!")
            continue
            
        print("\nGenerating answer...")
        start_time = time.time()
        answer = generate_answer(question, model, tokenizer, device)
        gen_time = time.time() - start_time
        print(f"\nAnswer: {answer}")
        print(f"[Generated in {gen_time:.2f} seconds]")

def main():
    # Configure environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    print("\n=== Optimized Medical Chatbot Evaluation ===")
    print("[INFO] This version uses mixed precision and batching for faster evaluation")
    print("[INFO] Updated with improved prompt format and generation parameters")
    
    # Check if we have a saved test set from training
    test_set_path = './HC_DATA/test_set.csv'
    has_test_set = os.path.exists(test_set_path)
    if has_test_set:
        print(f"[INFO] Found saved test set at {test_set_path} (recommended for consistent evaluation)")
    else:
        print("[INFO] No saved test set found, will use standard dataset")
    
    # Determine batch size based on available GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        default_batch_size = 8 if gpu_mem < 8 else 16
        print(f"[INFO] Detected {gpu_mem:.1f} GB GPU memory, using batch size {default_batch_size}")
    else:
        default_batch_size = 4
    
    # Get model path with improved suggestions
    print("\nAvailable model paths:")
    print("1. ./t5_chatbot_model/model (default, recommended)")
    print("2. ./t5_chatbot_model/checkpoint-1107")
    print("3. ./t5_chatbot_model/checkpoint-1000")
    print("4. ./t5_chatbot_model/peft_adapter")
    print("5. Custom path")
    
    model_choice = input("\nSelect model path (1-5): ").strip()
    
    if model_choice == "1" or not model_choice:
        model_path = "./t5_chatbot_model/model"
    elif model_choice == "2":
        model_path = "./t5_chatbot_model/checkpoint-1107"
    elif model_choice == "3":
        model_path = "./t5_chatbot_model/checkpoint-1000"
    elif model_choice == "4":
        model_path = "./t5_chatbot_model/peft_adapter"
    elif model_choice == "5":
        model_path = input("Enter custom model path: ").strip()
    else:
        model_path = "./t5_chatbot_model/model"
        print(f"Invalid choice, using default: {model_path}")
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    if model is None:
        return
    
    while True:
        print("\n=== Medical Chatbot Evaluation ===")
        print("1. Run model evaluation on test set")
        print("2. Interactive mode")
        print("3. Performance benchmark")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Evaluate model on test set
            print("\n=== Test Set Options ===")
            print("1. Use saved test set from training (if available)")
            print("2. Use MedQuad dataset")
            print("3. Use custom dataset")
            
            test_choice = input("\nSelect test data (1-3): ").strip()
            
            if test_choice == '1' and os.path.exists('./HC_DATA/test_set.csv'):
                test_file = "./HC_DATA/test_set.csv"
                print("[INFO] Using saved test set for consistent evaluation")
            elif test_choice == '2' or (test_choice == '1' and not os.path.exists('./HC_DATA/test_set.csv')):
                test_file = "./HC_DATA/medquad_qa_pairs.csv"
                print("[INFO] Using MedQuad dataset")
            else:
                test_file = input("Enter custom test data path: ").strip()
                if not os.path.exists(test_file):
                    print(f"[ERROR] File not found: {test_file}")
                    continue
            
            num_samples = input("Enter number of test samples (press Enter for all): ").strip()
            num_samples = int(num_samples) if num_samples.isdigit() else None
            
            # Use default number of samples if none specified
            if num_samples is None:
                num_samples = input("Enter number of test samples (press Enter for all): ").strip()
                num_samples = int(num_samples) if num_samples.isdigit() else None
                
            evaluate_model(model, tokenizer, device, test_data_path=test_file, num_samples=num_samples)
        
        elif choice == '2':
            interactive_mode(model, tokenizer, device)
        
        elif choice == '3':
            # Run performance benchmark
            print("\n=== Performance Benchmark ===")
            batch_size = input(f"Enter batch size for benchmark (press Enter for {default_batch_size}): ").strip()
            batch_size = int(batch_size) if batch_size.isdigit() else default_batch_size
            
            num_samples = input("Enter number of samples for benchmark (press Enter for 100): ").strip()
            num_samples = int(num_samples) if num_samples.isdigit() else 100
            
            print(f"[INFO] Running benchmark with batch size {batch_size} on {num_samples} samples...")
            start = time.time()
            
            # Create dummy dataset
            dummy_questions = ["What is diabetes?"] * num_samples
            
            # Process in batches
            for i in tqdm(range(0, len(dummy_questions), batch_size), desc="Benchmark"):
                batch = dummy_questions[i:i+batch_size]
                _ = generate_batch_answers(batch, model, tokenizer, device, batch_size=batch_size)
            
            benchmark_time = time.time() - start
            throughput = num_samples / benchmark_time
            
            print(f"\nBenchmark Results:")
            print(f"Total time: {benchmark_time:.2f} seconds")
            print(f"Throughput: {throughput:.2f} samples/second")
            print(f"Latency per sample: {1000 * benchmark_time / num_samples:.2f} ms")
        
        elif choice == '4':
            break
        
        else:
            print("Invalid choice! Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
# End of file