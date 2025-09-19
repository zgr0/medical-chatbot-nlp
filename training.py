
print("[INFO] Importing libraries...")
import torch  # PyTorch for deep learning
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

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")


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

