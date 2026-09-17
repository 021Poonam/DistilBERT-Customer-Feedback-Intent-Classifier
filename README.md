# DistilBERT Customer Feedback & Intent Classifier

A lightweight Natural Language Processing (NLP) classification pipeline built with **PyTorch** and Hugging Face **Transformers**. This project fine-tunes a pre-trained `distilbert-base-uncased` transformer model to categorize user feedback into four intent classes, providing real-time prediction confidence scores.

## Intent Classes
- `0: Bug Report`
- `1: Feature Request`
- `2: Positive Feedback`
- `3: Negative Feedback`

## Tech Stack
- **Language:** Python
- **Frameworks:** PyTorch, Hugging Face Transformers, Datasets, Evaluate
- **Model Architecture:** DistilBERT (`distilbert-base-uncased`)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
