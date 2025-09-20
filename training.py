from tqdm.auto import tqdm

print("[INFO] Importing libraries...")
# Set CUDA device settings for memory efficiency
import torch  # PyTorch for deep learning
torch.cuda.empty_cache()  # Clear GPU memory before starting
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner

# Set memory growth for better memory management
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass  # Ignore if TensorFlow is not properly installed
import numpy as np  # NumPy for numerical operations
import pandas as pd  # Pandas for data manipulation
import re  # Regular expressions for text processing
import tensorflow as tf  # TensorFlow (not used in this script, but imported)
import evaluate  # Library for evaluation metrics
import seaborn as sns  # Seaborn for data visualization
import matplotlib.pyplot as plt  # Matplotlib for plotting
import warnings  # To suppress warnings

# Import Hugging Face Transformers components
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, TrainerCallback, T5Config

# Import datasets for handling data
from datasets import Dataset

# Import scikit-learn for train-test split
from sklearn.model_selection import train_test_split

# Import PyTorch components for loss, optimization, and data handling
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import TensorDataset
from torchsummary import summary

# Import defaultdict for handling dictionaries with default values
from collections import defaultdict

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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


# Main execution logic
def main():
    print("[INFO] If you see a warning about 'tf.losses.sparse_softmax_cross_entropy', use 'tf.compat.v1.losses.sparse_softmax_cross_entropy' instead.")



    print("[INFO] Starting data loading...")
    dfs = []
    csv_files = [f for f in os.listdir('HC_DATA') if f.endswith('.csv')]
    print(f"[INFO] Found {len(csv_files)} CSV files in HC_DATA.")
    for filename in csv_files:
        file_path = os.path.join('HC_DATA', filename)
        print(f"[INFO] Loading file: {file_path}")
        dfs.append(pd.read_csv(file_path))
    print(f"[INFO] Total files loaded: {len(dfs)}")
    print("[INFO] Concatenating dataframes...")
    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total records after concatenation: {len(df)}")
    print("[INFO] Data Sample:")
    print(df.head())

    print("[INFO] Checking for null values in the dataset...")
    print(df.isnull().sum())

    question_words = ['what', 'who', 'why', 'when', 'where', 'how', 'is', 'are', 'does', 'do', 'can', 'will', 'shall']

    print("[INFO] Filtering questions by question words...")
    df['question'] = df['question'].str.lower()
    df = df[df['question'].str.split().str[0].isin(question_words)]
    print(f"[INFO] Records after filtering by question words: {len(df)}")
    df = df.reset_index(drop=True)

    duplicates = df.duplicated()
    print(f"[INFO] Number of duplicate rows: {duplicates.sum()}")
    df = df.drop_duplicates()
    print(f"[INFO] Records after dropping duplicates: {len(df)}")
    df.reset_index(drop=True, inplace=True)

    print("[INFO] Dropping unused columns: source, focus_area")
    df = df.drop(columns=['source', 'focus_area'])

    print("[INFO] Table Info:")
    print(df.info())

    print("[INFO] Dropping duplicates based on 'question' and 'answer' columns...")
    df = df.drop_duplicates(subset='question', keep='first').reset_index(drop=True)
    df = df.drop_duplicates(subset='answer', keep='first').reset_index(drop=True)

    print("[INFO] Dropping rows with null values in 'question' or 'answer' columns...")
    df = df.dropna(subset=['question', 'answer']).reset_index(drop=True)

    df['question'] = df['question'].fillna('').astype(str)
    df['answer'] = df['answer'].fillna('').astype(str)

    print("[INFO] Cleaning text in 'question' and 'answer' columns...")
    def clean_text(text):
        text = re.sub(r"\(.*?\)", "", text)  # Remove text within parentheses
        text = re.sub(r'\s+', ' ', text.strip().lower())  # Normalize spaces and convert to lowercase
        return text
    df['question'] = df['question'].apply(clean_text)
    df['answer'] = df['answer'].apply(clean_text)
    df['question'] = df['question'].str.lower().str.strip().apply(lambda x: re.sub(r'\s+', ' ', x))
    df['answer'] = df['answer'].str.lower().str.strip().apply(lambda x: re.sub(r'\s+', ' ', x))

    print("[INFO] Null Value Data After Cleaning:")
    print(df.isnull().sum())

    print(f"[INFO] Unique questions: {df['question'].nunique()}")
    print(f"[INFO] Unique answers: {df['answer'].nunique()}")

    print("[INFO] Final Dataset Info:")
    df.info()
    print("[INFO] Final Data Sample:")
    print(df.head())
    print("[INFO] Data preprocessing complete. Ready for training setup.")

    # Define the model name and load the T5 configuration
    model_name = "t5-base"
    config = T5Config.from_pretrained(model_name)

    # Customize the configuration
    config.dropout_rate = 0.1  # Set dropout rate to 0.1 for regularization
    config.feed_forward_proj = "gelu"  # Use GELU activation for the feed-forward layers

    # Load the pre-trained T5 model with the customized configuration
    model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)

    # Load the tokenizer for the T5 model
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Explicitly resize the token embeddings to match the tokenizer's vocabulary size
    model.resize_token_embeddings(len(tokenizer))

    # Print a detailed summary of the model architecture
    print("\nDetailed Model Summary:")
    print("=" * 50)

    def summarize_model_by_type(model):
        """
        Summarizes the model by counting the number of layers and parameters for each layer type.
        """
        layer_summary = defaultdict(int)  # Counts the number of layers by type
        param_summary = defaultdict(int)  # Counts the number of parameters by layer type

        for name, module in model.named_modules():
            layer_type = type(module).__name__  # Get the type of the current module
            layer_summary[layer_type] += 1  # Increment the count for this layer type
            param_summary[layer_type] += sum(p.numel() for p in module.parameters())  # Sum parameters

        # Print the summary table
        print(f"{'Layer Type':<30}{'Count':<10}{'Parameters':<15}")
        print("=" * 55)
        for layer_type, count in layer_summary.items():
            print(f"{layer_type:<30}{count:<10}{param_summary[layer_type]:<15,}")

    summarize_model_by_type(model)

    # Define a preprocessing function for the seq2seq task
    def preprocess_function(batch):
        """
        Preprocesses the dataset by tokenizing the inputs and targets.
        """
        # Format the inputs and targets
        inputs = [f"answer the following question: {q}" for q in batch['question']]
        targets = [f"{a}" for a in batch['answer']]

        # Tokenize the inputs
        model_inputs = tokenizer(
            inputs,
            max_length=128,  # Truncate or pad to a maximum length of 128
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize the targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=64,  # Truncate or pad to a maximum length of 64
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        # Replace padding token IDs with -100 for the loss function to ignore them
        labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    # Convert the pandas DataFrames to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Preprocess the training and validation datasets
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=16,  # Reduced preprocessing batch size
        remove_columns=train_dataset.column_names,  # Remove original columns
        num_proc=2,  # Reduced number of processes for laptop efficiency
    )

    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=16,  # Reduced preprocessing batch size
        remove_columns=val_dataset.column_names,  # Remove original columns
        num_proc=2,  # Reduced number of processes for laptop efficiency
    )

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",  # Directory to save the model and results
        evaluation_strategy="epoch",  # Evaluate after each epoch
        save_total_limit=2,  # Keep only the last 2 checkpoints
        learning_rate=5e-4,  # Learning rate
        num_train_epochs=5,  # Number of training epochs
        per_device_train_batch_size=4,  # Reduced batch size for training
        per_device_eval_batch_size=4,  # Reduced batch size for evaluation
        lr_scheduler_type="cosine_with_restarts",  # Learning rate scheduler
        warmup_ratio=0.1,  # Warmup ratio for the scheduler
        weight_decay=0.05,  # Weight decay for regularization
        predict_with_generate=True,  # Generate predictions during evaluation
        fp16=True,  # Use mixed precision for faster training
        bf16=False,  # Disable bfloat16 as it's not optimal for RTX 4060
        logging_dir="./logs",  # Directory for logs
        logging_steps=50,  # Log every 50 steps
        metric_for_best_model="exact_match",  # Use exact match as the primary metric
        greater_is_better=True,  # Higher exact match is better
        report_to="none",  # Disable external reporting
        gradient_accumulation_steps=4,  # Increased gradient accumulation for memory efficiency
        max_grad_norm=0.5,  # Gradient clipping
        optim="adamw_torch_fused",  # Use fused AdamW optimizer
        generation_max_length=64,  # Maximum length for generated text
        generation_num_beams=6,  # Number of beams for beam search
        dataloader_num_workers=2,  # Reduced number of workers for laptop efficiency
        dataloader_pin_memory=True,  # Enable pin memory for faster data transfer
        group_by_length=True,  # Group sequences by length for efficiency
        remove_unused_columns=True,  # Remove unused columns from the dataset
        label_smoothing_factor=0.1,  # Apply label smoothing
    )

    # Initialize the data collator for seq2seq tasks
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding='longest',  # Pad sequences to the longest in the batch
        return_tensors="pt",  # Return PyTorch tensors
    )

    # Define a function to compute evaluation metrics
    def compute_metrics(eval_pred, tokenizer):
        """
        Computes exact match, BLEU, and ROUGE-L metrics for evaluation.
        """
        predictions, labels = eval_pred

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Normalize text for comparison
        decoded_preds = [text.strip().lower() for text in decoded_preds]
        decoded_labels = [text.strip().lower() for text in decoded_labels]

        # Compute exact match
        exact_match = np.mean([p == l for p, l in zip(decoded_preds, decoded_labels)])

        # Load BLEU and ROUGE metrics
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")

        # Compute BLEU score
        bleu_score = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )["bleu"]

        # Compute ROUGE-L score
        rouge_score = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )["rougeL"]

        return {
            "exact_match": exact_match,
            "BLEU": bleu_score,
            "ROUGE-L": rouge_score,
        }

    # Initialize the Seq2SeqTrainer

    # Add the progress bar callback to the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        callbacks=[TQDMProgressBarCallback()],
    )

    # Train the model
    trainer.train()

    # Save the trained model and tokenizer
    trainer.save_model("./t5_chatbot_model")
    tokenizer.save_pretrained("./t5_chatbot_tokenizer")

    # Save the model's state dictionary
    model_path = "./t5_chatbot_model.h5"
    torch.save(model.state_dict(), model_path)

    # Save the training log history
    log_history = trainer.state.log_history


if __name__ == "__main__":
    main()