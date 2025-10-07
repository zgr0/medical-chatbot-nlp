import os
import sys
from tqdm.auto import tqdm
import torch
import pandas as pd
import numpy as np
import evaluate
import warnings

# Import Hugging Face Transformers components
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments,
    Seq2SeqTrainer, DataCollatorForSeq2Seq, TrainerCallback, T5Config
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")

# Custom callback to show a tqdm progress bar during training
class TQDMProgressBarCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None
        self.total_steps = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps
        self.pbar = tqdm(total=self.total_steps, desc="Training Progress", unit="step")

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.close()

def main():
    """
    Main function to train a medical chatbot model using prepared data
    with improved output length handling
    """
    print("\n" + "="*50)
    print(" IMPROVED MEDICAL CHATBOT TRAINING")
    print("="*50)
    
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set environment variables for PyTorch performance
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Get environment variables for training configuration
    # Use default values if environment variables are not set
    output_dir = os.environ.get("OUTPUT_DIR", "./t5_chatbot_model")
    num_epochs = int(os.environ.get("NUM_EPOCHS", "3"))
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    
    # Get environment variables for mixed precision
    use_fp16 = os.environ.get("USE_FP16", "true").lower() == "true"
    
    # Get environment variable for parameter-efficient fine-tuning method
    peft_method = os.environ.get("PEFT_METHOD", "lora").lower()
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("[INFO] Configuration:")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - FP16 mixed precision: {use_fp16}")
    print(f"  - PEFT method: {peft_method}")
    print(f"  - PyTorch version: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load the prepared data
    processed_dir = os.path.join('HC_DATA', 'processed')
    
    if not os.path.exists(processed_dir):
        print(f"[ERROR] Processed data directory not found: {processed_dir}")
        print("[INFO] Please run process_data.py first to prepare the data")
        return
        
    try:
        train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(processed_dir, 'val.csv'))
        print(f"[INFO] Loaded data: {len(train_df)} training samples, {len(val_df)} validation samples")
        
        # Print statistics about answer lengths
        answer_lengths = [len(str(a).split()) for a in train_df['answer']]
        print(f"[INFO] Answer length statistics:")
        print(f"  - Minimum: {min(answer_lengths)} words")
        print(f"  - Maximum: {max(answer_lengths)} words")
        print(f"  - Average: {sum(answer_lengths)/len(answer_lengths):.1f} words")
        print(f"  - 95th percentile: {np.percentile(answer_lengths, 95):.1f} words")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return
    
    # Define the model name and load the T5 configuration
    model_name = "t5-base"
    print(f"[INFO] Loading model configuration: {model_name}")
    
    # Define enhanced T5 configuration for medical domain
    config = T5Config.from_pretrained(model_name)
    config.dropout_rate = 0.1  # Moderate dropout to prevent overfitting
    
    # Additional medical domain-specific configuration
    print("[INFO] Configuring model with medical domain specializations")
    
    # Load the pre-trained T5 model with careful initialization
    print(f"[INFO] Loading model weights: {model_name}")
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, 
        config=config,
        return_dict=True
    )
    
    # Set up generation config with optimized parameters for medical QA
    print("[INFO] Configuring generation parameters for medical responses")
    model.config.repetition_penalty = 2.0       # Stronger penalty for repetition
    model.config.no_repeat_ngram_size = 3       # Avoid repeating 3-grams
    model.config.early_stopping = False         # Generate complete answers
    model.config.length_penalty = 2.0           # Strong preference for longer answers
    model.config.max_length = 512               # Allow for long medical explanations
    model.config.min_length = 100               # Ensure comprehensive answers
    model.config.diversity_penalty = 0.5        # Encourage diverse content
    model.config.num_beam_groups = 4            # Use beam groups for diverse outputs
    model.config.temperature = 0.8              # Slightly reduced randomness
    
    print("[INFO] Model configuration complete")
    
    # Apply parameter-efficient fine-tuning if specified
    if peft_method == "lora":
        print("[INFO] Applying LoRA parameter-efficient fine-tuning with enhanced settings")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=64,                     # Further increased low-rank dimension for better medical adaptation
            lora_alpha=128,           # Increased alpha scaling factor for stronger updates
            lora_dropout=0.1,         # Increased dropout for better generalization
            bias="none",              # Don't train biases to stabilize training
            # Apply LoRA to more modules for comprehensive parameter adaptation
            target_modules=["q", "v", "k", "o", "wi", "wo"], 
            modules_to_save=["embed_tokens", "lm_head"],  # Save critical components for better prediction
            fan_in_fan_out=False      # T5 convention
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("[INFO] Model moved to CUDA")

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    print("[INFO] Tokenizer loaded")

    # Define enhanced preprocessing function
    def preprocess_function(batch):
        """
        Enhanced preprocessing of dataset by tokenizing inputs and targets,
        with improved prompting for medical answers
        """
        # Format inputs with a clearer, more structured prompt to guide the model better
        inputs = [f"Medical question: {q.strip()}\n\nProvide a comprehensive, accurate, and detailed medical answer. Include medical terminology, causes, symptoms, treatments, and recommendations:" for q in batch['question']]
        
        # Clean and format targets for better learning
        targets = [f"{a.strip()}" for a in batch['answer']]
        
        # Print sample examples for inspection
        if not hasattr(preprocess_function, 'samples_shown'):
            print("\n[INFO] Sample training examples:")
            for i in range(min(3, len(batch['question']))):
                print(f"Input: {inputs[i]}")
                print(f"Target: {targets[i][:150]}..." if len(targets[i]) > 150 else f"Target: {targets[i]}")
                print()
            preprocess_function.samples_shown = True

        # Tokenize inputs with increased max_length for detailed prompts
        model_inputs = tokenizer(
            inputs,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize targets with larger max_length for comprehensive medical answers
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=512,  # Further increased for very comprehensive medical answers
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        # Replace padding token IDs with -100 for loss calculation
        labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Convert DataFrames to Hugging Face Datasets
    print("[INFO] Converting data to Hugging Face datasets")
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Preprocess datasets
    print("[INFO] Preprocessing training dataset")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=32,
        remove_columns=train_dataset.column_names,
        desc="Processing training dataset"
    )

    print("[INFO] Preprocessing validation dataset")
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=32,
        remove_columns=val_dataset.column_names,
        desc="Processing validation dataset"
    )

    # Define optimized training arguments for medical text generation
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        learning_rate=3e-5,             # Slightly reduced for more stable learning
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,   # Small eval batch size for memory efficiency
        warmup_ratio=0.15,              # Increased warmup for better convergence
        weight_decay=0.05,              # Increased for better regularization
        predict_with_generate=True,
        fp16=use_fp16,
        generation_max_length=512,      # Back to full length for proper evaluation
        generation_num_beams=4,         # More beams for better quality
        logging_steps=50,
        metric_for_best_model="rouge2",  # Use ROUGE-2 which is better for medical content
        greater_is_better=True,
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,
        adafactor=True,                 # Use Adafactor optimizer for memory efficiency
        length_penalty=2.0,             # Favor longer outputs
        no_repeat_ngram_size=3,         # Avoid repetition
        lr_scheduler_type="cosine",     # Use cosine scheduler for better convergence
        group_by_length=True,           # Group similar-length sequences for efficiency
        report_to=["tensorboard"],      # Use tensorboard for monitoring
        seed=42,                        # Fix seed for reproducibility
        label_smoothing_factor=0.1      # Add label smoothing for better generalization
    )

    # Initialize data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding='longest',
        return_tensors="pt",
    )

    # Define comprehensive metrics function for medical text evaluation
    def compute_metrics(eval_pred):
        """Comprehensive metrics computation for medical text generation"""
        print("\n[INFO] Running enhanced metrics computation")
        predictions, labels = eval_pred
        
        try:
            # Decode predictions
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            # Replace -100 with pad token ID for decoding
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Text normalization optimized for medical content
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]
            
            # Print sample predictions for detailed inspection
            if len(decoded_preds) > 0:
                print("\n[INFO] Sample prediction from evaluation:")
                for i in range(min(2, len(decoded_preds))):
                    print(f"Prediction ({len(decoded_preds[i].split())} words): {decoded_preds[i][:150]}...")
                    print(f"Reference ({len(decoded_labels[i].split())} words): {decoded_labels[i][:150]}...")
                    print()
            
            # Load metrics modules
            rouge = evaluate.load("rouge")
            bleu = evaluate.load("bleu")
            
            # Compute ROUGE with comprehensive settings
            rouge_output = rouge.compute(
                predictions=decoded_preds, 
                references=decoded_labels,
                use_stemmer=True,
                use_aggregator=True
            )
            
            # Compute BLEU
            bleu_output = bleu.compute(
                predictions=decoded_preds,
                references=[[ref] for ref in decoded_labels]
            )
            
            # Exact match and length-based metrics
            exact_match = sum(1 for pred, ref in zip(decoded_preds, decoded_labels) 
                             if pred.lower() == ref.lower()) / max(len(decoded_preds), 1)
            
            # Calculate length statistics
            pred_lengths = [len(pred.split()) for pred in decoded_preds]
            ref_lengths = [len(ref.split()) for ref in decoded_labels]
            avg_pred_length = sum(pred_lengths) / max(len(pred_lengths), 1)
            avg_ref_length = sum(ref_lengths) / max(len(ref_lengths), 1)
            length_ratio = avg_pred_length / max(avg_ref_length, 1)
            
            # Return comprehensive metrics
            metrics = {
                "rouge1": rouge_output["rouge1"],
                "rouge2": rouge_output["rouge2"],
                "rougeL": rouge_output["rougeL"],
                "bleu": bleu_output["bleu"],
                "exact_match": exact_match,
                "avg_pred_length": avg_pred_length,
                "avg_ref_length": avg_ref_length,
                "length_ratio": length_ratio
            }
            
            print(f"[INFO] Evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            print(f"[ERROR] Metrics computation error: {e}")
            import traceback
            traceback.print_exc()
            return {"rouge2": 0.0}

    # Create a representative validation dataset for evaluation
    print("[INFO] Creating evaluation dataset")
    # Use more validation samples for better evaluation
    eval_dataset = val_dataset.select(range(min(50, len(val_dataset))))
    
    # Custom callback for enhanced training monitoring
    class EnhancedTrainingCallback(TrainerCallback):
        """Enhanced callback with more detailed monitoring and logging"""
        def __init__(self):
            self.best_metric = None
            self.best_step = 0
            self.training_start_time = None
            
        def on_train_begin(self, args, state, control, **kwargs):
            import time
            self.training_start_time = time.time()
            print("[INFO] Beginning training with enhanced monitoring...")
        
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 50 == 0:
                import time
                elapsed = time.time() - self.training_start_time
                steps_per_second = state.global_step / max(elapsed, 1)
                remaining_steps = state.max_steps - state.global_step
                eta_seconds = remaining_steps / max(steps_per_second, 1e-6)
                
                print(f"[INFO] Step {state.global_step}/{state.max_steps} " +
                      f"| Loss: {state.log_history[-1].get('loss', 'N/A'):.4f} " +
                      f"| {steps_per_second:.2f} steps/s | ETA: {eta_seconds/60:.1f}m")
                
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            print(f"\n[INFO] Evaluation at step {state.global_step} completed")
            
            if metrics and "rouge2" in metrics:
                current = metrics["rouge2"]
                if self.best_metric is None or current > self.best_metric:
                    self.best_metric = current
                    self.best_step = state.global_step
                    print(f"[INFO] New best ROUGE-2 score: {current:.4f}")
                print(f"[INFO] Best score so far: {self.best_metric:.4f} at step {self.best_step}")
                
        def on_train_end(self, args, state, control, **kwargs):
            import time
            total_time = time.time() - self.training_start_time
            print(f"\n[INFO] Training completed in {total_time/60:.2f} minutes")
            print(f"[INFO] Best ROUGE-2 score: {self.best_metric:.4f} at step {self.best_step}")

    # Apply model optimizations before training
    print("[INFO] Applying additional model optimizations")
    
    # Create optimizer with learning rate schedule
    from transformers import Adafactor
    optimizer = Adafactor(
        model.parameters(),
        lr=training_args.learning_rate,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False
    )
    
    # Initialize enhanced trainer
    print("[INFO] Initializing trainer with advanced configuration")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Use expanded evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EnhancedTrainingCallback()],
        optimizers=(optimizer, None)  # Use custom optimizer
    )

    # Train with a try-except block to handle any errors
    print("[INFO] Starting training")
    try:
        trainer.train()
        print("[INFO] Training completed successfully!")
    except Exception as e:
        print(f"[ERROR] Training error: {e}")
        print("[INFO] Saving model despite error...")
    
    # Save all model components with proper error handling
    try:
        print(f"[INFO] Saving model components to {output_dir}")
        
        # Save PEFT adapter if using parameter-efficient fine-tuning
        if peft_method == "lora":
            peft_adapter_dir = os.path.join(output_dir, "peft_adapter")
            os.makedirs(peft_adapter_dir, exist_ok=True)
            model.save_pretrained(peft_adapter_dir)
            print(f"[INFO] LoRA adapter saved to {peft_adapter_dir}")
        
        # Save the model
        model_dir = os.path.join(output_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        trainer.save_model(model_dir)
        print(f"[INFO] Model saved to {model_dir}")
        
        # Save tokenizer
        tokenizer_dir = os.path.join(output_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_dir)
        print(f"[INFO] Tokenizer saved to {tokenizer_dir}")
        
        print("[INFO] All model components saved successfully!")
        
    except Exception as e:
        print(f"[ERROR] Failed to save some model components: {e}")
    
    print("[INFO] Training process complete!")

if __name__ == "__main__":
    # Set environment flags for testing if needed
    if "--fp16" in sys.argv:
        os.environ["USE_FP16"] = "true"
    if "--fp32" in sys.argv:
        os.environ["USE_FP16"] = "false"
    if "--lora" in sys.argv:
        os.environ["PEFT_METHOD"] = "lora"
    if "--full" in sys.argv:
        os.environ["PEFT_METHOD"] = "none"
    
    main()