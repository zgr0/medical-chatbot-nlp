# Medical Chatbot NLP

This project is a medical question-answering chatbot built using natural language processing (NLP) techniques and deep learning. It leverages the T5 transformer model to generate answers to medical questions, with a focus on data cleaning, preprocessing, and robust evaluation.

## Features
- Cleans and preprocesses medical Q&A data
- Trains a T5-based sequence-to-sequence model for question answering
- Includes progress bar and verbose logging for training
- Evaluates model performance using metrics such as Exact Match, BLEU, and ROUGE-L

## Dataset
The primary dataset used for training and evaluation is sourced from Kaggle:

[https://www.kaggle.com/datasets/jpmiller/layoutlm](https://www.kaggle.com/datasets/jpmiller/layoutlm)

## Requirements
- Python 3.8+
- PyTorch (with CUDA support recommended)
- Hugging Face Transformers
- Datasets, Evaluate, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, tqdm
- TensorFlow (for compatibility)

## Usage
1. Install the required dependencies (see `training.py` for details).
2. Download and extract the dataset from the Kaggle link above.
3. Run the training script:
   ```bash
   python training.py
   ```
4. The script will preprocess the data, train the model, and save the trained weights and tokenizer.

## Notes
- The script is designed for research and educational purposes.
- For best results, ensure your environment has a compatible GPU and CUDA drivers installed.