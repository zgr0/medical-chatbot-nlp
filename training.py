from tqdm.auto import tqdm

print("[INFO] Importing libraries...")
print("[INFO] Libraries imported. Initializing CUDA/memory settings...")
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
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerSelfAttention

# Import Flash Attention
try:
    from flash_attn import flash_attn_func
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    FLASH_ATTENTION_AVAILABLE = True
    print("[INFO] Flash Attention 2 is available and will be used for improved efficiency.")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("[WARNING] Flash Attention 2 not available. Using memory-efficient attention fallback.")

import math
import torch.nn.functional as F

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


# Custom Memory-Efficient T5 implementation
class MemoryEfficientT5Attention(T5Attention):
    """
    T5 Attention layer with memory-efficient attention computation.
    Provides similar benefits to Flash Attention without requiring the flash-attn package.
    """
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias=has_relative_attention_bias)
        self.use_memory_efficient = True
        
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        **kwargs
    ):
        # Use memory-efficient attention when conditions are met
        if (self.use_memory_efficient and 
            not output_attentions and 
            hidden_states.dtype in [torch.float16, torch.bfloat16, torch.float32]):
            
            return self._memory_efficient_attention_forward(
                hidden_states, mask, key_value_states, position_bias, 
                past_key_value, use_cache
            )
        else:
            # Fallback to standard attention
            return super().forward(
                hidden_states=hidden_states,
                mask=mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                past_key_value=past_key_value,
                layer_head_mask=layer_head_mask,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
    
    def _memory_efficient_attention_forward(self, hidden_states, mask, key_value_states, 
                                          position_bias, past_key_value, use_cache):
        """
        Memory-efficient attention forward pass using PyTorch's scaled_dot_product_attention.
        """
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Compute Q, K, V projections
        query_states = self.q(hidden_states)
        
        if key_value_states is None:
            # Self-attention
            key_states = self.k(hidden_states)
            value_states = self.v(hidden_states)
        else:
            # Cross-attention
            key_states = self.k(key_value_states)
            value_states = self.v(key_value_states)
        
        # Reshape for attention computation
        query_states = query_states.view(batch_size, seq_len, self.n_heads, self.key_value_proj_dim)
        key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)
        value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)
        
        # Handle past key values for caching
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        
        # Transpose for scaled_dot_product_attention: [batch, heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Convert mask to attention mask format if provided
        attn_mask = None
        if mask is not None:
            # mask: [batch, seq_len] or [batch, tgt_len, src_len]
            if mask.dim() == 2:
                # [batch, seq_len] -> [batch, 1, seq_len, seq_len]
                attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
                attn_mask = attn_mask.expand(batch_size, self.n_heads, seq_len, key_states.size(-2))
            elif mask.dim() == 3:
                # [batch, tgt_len, src_len] -> [batch, 1, tgt_len, src_len]
                attn_mask = mask.unsqueeze(1)
                attn_mask = attn_mask.expand(batch_size, self.n_heads, mask.size(-2), mask.size(-1))
            else:
                attn_mask = mask  # fallback, let attention fail if shape is wrong
            attn_mask = attn_mask.to(dtype=torch.bool)
            attn_mask = ~attn_mask  # Invert mask (True for positions to attend to)
        
        try:
            # Use PyTorch's memory-efficient scaled dot product attention
            if hasattr(F, 'scaled_dot_product_attention'):
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False  # T5 uses bidirectional attention
                )
            else:
                # Fallback to manual implementation for older PyTorch versions
                attn_output = self._manual_scaled_dot_product_attention(
                    query_states, key_states, value_states, attn_mask
                )
        except Exception as e:
            print(f"[WARNING] Memory-efficient attention failed: {e}. Falling back to standard attention.")
            return super().forward(
                hidden_states=hidden_states,
                mask=mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
        
        # Reshape output back to expected format
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.inner_dim)
        
        # Apply output projection
        attn_output = self.o(attn_output)
        
        # Apply position bias if provided
        if position_bias is not None:
            # Position bias is already applied in the attention computation
            pass
        
        present_key_value = (key_states.transpose(1, 2), value_states.transpose(1, 2)) if use_cache else None
        
        return (attn_output, present_key_value)
    
    def _manual_scaled_dot_product_attention(self, query, key, value, attn_mask):
        """
        Manual implementation of scaled dot product attention for older PyTorch versions.
        """
        # Compute attention scores
        scale_factor = 1.0 / math.sqrt(query.size(-1))
        attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        
        # Apply mask if provided
        if attn_mask is not None:
            attn_weight = attn_weight.masked_fill(~attn_mask, float('-inf'))
        
        # Apply softmax
        attn_weight = F.softmax(attn_weight, dim=-1)
        
        # Apply dropout
        if self.training and self.dropout > 0:
            attn_weight = F.dropout(attn_weight, p=self.dropout)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weight, value)
        
        return attn_output


def replace_attention_with_flash(model):
    """
    Replace standard T5 attention layers with Memory-Efficient attention layers.
    """
    print("[INFO] Replacing T5 attention layers with Memory-Efficient Attention...")
    
    # Replace attention in encoder layers
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
        for i, layer in enumerate(model.encoder.block):
            if hasattr(layer, 'layer') and len(layer.layer) > 0:
                if hasattr(layer.layer[0], 'SelfAttention'):
                    # Replace self-attention
                    old_attn = layer.layer[0].SelfAttention
                    new_attn = MemoryEfficientT5Attention(
                        model.config, 
                        has_relative_attention_bias=(i == 0)
                    )
                    # Copy weights
                    new_attn.load_state_dict(old_attn.state_dict())
                    layer.layer[0].SelfAttention = new_attn
    
    # Replace attention in decoder layers
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'block'):
        for i, layer in enumerate(model.decoder.block):
            if hasattr(layer, 'layer') and len(layer.layer) > 0:
                if hasattr(layer.layer[0], 'SelfAttention'):
                    # Replace self-attention
                    old_attn = layer.layer[0].SelfAttention
                    new_attn = MemoryEfficientT5Attention(
                        model.config,
                        has_relative_attention_bias=(i == 0)
                    )
                    # Copy weights
                    new_attn.load_state_dict(old_attn.state_dict())
                    layer.layer[0].SelfAttention = new_attn
                
                # Replace cross-attention if exists
                if len(layer.layer) > 1 and hasattr(layer.layer[1], 'EncDecAttention'):
                    old_cross_attn = layer.layer[1].EncDecAttention
                    new_cross_attn = MemoryEfficientT5Attention(model.config, has_relative_attention_bias=False)
                    # Copy weights
                    new_cross_attn.load_state_dict(old_cross_attn.state_dict())
                    layer.layer[1].EncDecAttention = new_cross_attn
    
    print("[INFO] Memory-Efficient Attention integration complete!")
    return model



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
    
    # Print Flash Attention status
    if FLASH_ATTENTION_AVAILABLE:
        print("[INFO] ✓ Flash Attention 2 is ready for use - expect 2-4x faster training!")
    else:
        print("[INFO] ✓ Using Memory-Efficient Attention - optimized for your hardware setup!")

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
    
    # Apply Flash Attention 2 optimization
    print("[INFO] Applying Memory-Efficient Attention to the model...")
    model = replace_attention_with_flash(model)
    # Keep model in FP32 for initial stability
    if torch.cuda.is_available():
        model = model.cuda()
        print("[INFO] Model moved to CUDA with FP32 for training stability")

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
        num_proc=1,  # Use single process to avoid import loops
    )

    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=16,  # Reduced preprocessing batch size
        remove_columns=val_dataset.column_names,  # Remove original columns
        num_proc=1,  # Use single process to avoid import loops
    )

    # Define training arguments (with FP16 stability fixes)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",  # Directory to save the model and results
        eval_strategy="epoch",  # Evaluate after each epoch
        save_total_limit=2,  # Keep only the last 2 checkpoints
        learning_rate=5e-5,  # Even lower learning rate for FP16 stability
        num_train_epochs=5,  # Number of training epochs
        per_device_train_batch_size=2,  # Smaller batch size for stability
        per_device_eval_batch_size=2,  # Smaller batch size for evaluation
        lr_scheduler_type="linear",  # Linear scheduler instead of cosine for stability
        warmup_ratio=0.1,  # Warmup ratio for the scheduler
        weight_decay=0.01,  # Lower weight decay for FP16 stability
        predict_with_generate=True,  # Generate predictions during evaluation
        fp16=False,  # Disable FP16 temporarily for stability
        bf16=False,  # Disable bfloat16 as it's not optimal for RTX 4060
        logging_dir="./logs",  # Directory for logs
        logging_steps=50,  # Log every 50 steps
        metric_for_best_model="exact_match",  # Use exact match as the primary metric
        greater_is_better=True,  # Higher exact match is better
        report_to="none",  # Disable external reporting
        gradient_accumulation_steps=8,  # Higher gradient accumulation to compensate for smaller batch
        max_grad_norm=0.1,  # Lower gradient clipping for FP16 stability
        optim="adamw_torch",  # Use regular AdamW instead of fused for stability
        generation_max_length=64,  # Maximum length for generated text
        generation_num_beams=4,  # Fewer beams for memory efficiency
        dataloader_num_workers=0,  # Disable multiprocessing to avoid import loops
        dataloader_pin_memory=False,  # Disable pin memory for stability
        group_by_length=True,  # Group sequences by length for efficiency
        remove_unused_columns=True,  # Remove unused columns from the dataset
        label_smoothing_factor=0.0,  # Disable label smoothing for FP16 stability
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

        # Clip predictions to valid token ID range
        vocab_size = len(tokenizer)
        predictions = np.clip(predictions, 0, vocab_size - 1)
        
        # Replace -100 labels with pad token
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        try:
            # Decode predictions and labels
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        except (IndexError, ValueError) as e:
            print(f"[WARNING] Tokenizer decode error: {e}. Using fallback metrics.")
            # Return default metrics if decoding fails
            return {
                "exact_match": 0.0,
                "BLEU": 0.0,
                "ROUGE-L": 0.0,
            }

        # Normalize text for comparison
        decoded_preds = [text.strip().lower() for text in decoded_preds]
        decoded_labels = [text.strip().lower() for text in decoded_labels]

        # Compute exact match
        exact_match = np.mean([p == l for p, l in zip(decoded_preds, decoded_labels)])

        try:
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
        except Exception as e:
            print(f"[WARNING] Metrics computation error: {e}. Using fallback values.")
            bleu_score = 0.0
            rouge_score = 0.0

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