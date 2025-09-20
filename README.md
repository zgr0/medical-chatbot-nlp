# Medical Question-Answering Chatbot with NLP

A sophisticated medical question-answering system built using state-of-the-art NLP techniques and transformer architecture. This project implements a T5-based sequence-to-sequence model to provide accurate and contextually relevant answers to medical questions.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Performance](#performance)
- [Contributing](#contributing)

## Overview

This project aims to create a reliable medical question-answering system that can understand and respond to various medical queries. It utilizes the T5 transformer model, fine-tuned on a comprehensive medical Q&A dataset, with careful attention to data preprocessing and model evaluation.

## Architecture

- **Base Model**: T5-base transformer architecture
- **Training Approach**: Fine-tuning with sequence-to-sequence learning
- **Optimization**: Mixed precision training (FP16) with gradient accumulation
- **Memory Efficiency**: Optimized for consumer GPUs (e.g., RTX 4060)

## Features

### Data Processing
- Robust text cleaning and normalization
- Question filtering and validation
- Duplicate removal and data quality checks
- Efficient batch processing

### Training
- Mixed precision training (FP16)
- Gradient accumulation for stable training
- Cosine learning rate scheduling
- Comprehensive logging and checkpointing

### Evaluation
- Multiple evaluation metrics (BLEU, ROUGE-L, Exact Match)
- Interactive testing interface
- Batch evaluation capabilities
- Detailed performance analytics

## Dataset

The project uses the LayoutLM medical dataset from Kaggle:
[https://www.kaggle.com/datasets/jpmiller/layoutlm](https://www.kaggle.com/datasets/jpmiller/layoutlm)

Key dataset characteristics:
- High-quality medical Q&A pairs
- Diverse medical topics coverage
- Professionally curated content
- Multiple medical domains

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-chatbot-nlp.git
cd medical-chatbot-nlp
```

2. Install required dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn transformers datasets evaluate torchsummary
```

## Usage

### Training
```bash
python training.py
```

### Evaluation
```bash
python evaluate_model.py
```

Select from:
1. Run evaluation on test set
2. Interactive mode for custom questions
3. Exit

## Training

### Configuration
- Learning Rate: 5e-4
- Batch Size: 4 (effective batch size: 16 with gradient accumulation)
- Training Epochs: 5
- Gradient Accumulation Steps: 4
- Mixed Precision: FP16

### Optimization
The training script is optimized for laptop GPUs with:
- Memory-efficient batching
- Gradient accumulation
- Pin memory for faster data transfer
- Optimized worker processes

## Evaluation

The evaluation script provides:
- Automatic GPU detection and utilization
- Multiple evaluation metrics
- Interactive testing mode
- Batch processing capabilities
- Detailed performance reporting

### Metrics
- BLEU Score
- ROUGE-L Score
- Exact Match Rate
- Response Generation Time

## Performance

(Add your model's performance metrics here after training)
- Test Set Accuracy: XX%
- BLEU Score: XX
- ROUGE-L Score: XX
- Average Response Time: XXms

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request