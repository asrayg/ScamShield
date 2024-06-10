# ScamShield: Real-Time Scam Detection System

## Abstract

ScamShield is a comprehensive real-time scam detection system designed to monitor phone calls, transcribe audio, and detect potential scam attempts. The system leverages advanced machine learning models for text classification, integrated within a robust backend, and provides an intuitive user interface through a desktop application. ScamShield aims to enhance user safety by identifying suspicious activities during phone calls, ensuring prompt alerts and actions.

## Table of Contents

1. [Model Description](#model-description)
2. [Backend Setup](#backend-setup)
3. [Frontend Setup](#frontend-setup)
4. [Running the Application](#running-the-application)

## Model Description

### Overview

The core of ScamShield is built on a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model. BERT is a state-of-the-art transformer-based model for natural language processing tasks. In ScamShield, we use BERT to classify transcribed audio from phone calls as either "spam" (potential scam) or "not_spam" (legitimate).

### Dataset

The model is trained on a custom dataset containing phone call transcripts labeled as "spam" (1) or "not_spam" (0). The dataset structure is as follows:

| text                                     | spam |
|------------------------------------------|------|
| "You have won a lottery. Click here..."  | 1    |
| "Please update your account information" | 1    |
| "Meeting is scheduled at 10 AM tomorrow."| 0    |
| "Your order has been shipped..."         | 0    |

### Training

The model is fine-tuned using the `transformers` library. We use the `BertForSequenceClassification` class for binary classification and train it on the custom dataset to achieve high accuracy in scam detection.

### Code Example

Here's a brief example of how the model is fine-tuned:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load and preprocess dataset
df = pd.read_csv('spam_dataset.csv')
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'].tolist(), df['spam'].tolist(), test_size=0.2)

# Tokenize the texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Create Dataset objects
train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': train_labels})
test_dataset = Dataset.from_dict({'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask'], 'labels': test_labels})

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('scam_classifier')
tokenizer.save_pretrained('scam_classifier')
```

## Backend Setup

### Overview

The backend is built using Django and Django REST Framework. It provides an API for analyzing transcriptions and detecting potential scams. The backend leverages the fine-tuned BERT model for classification.

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/scamshield.git
   cd scamshield
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** in the project root with the following content:
   ```plaintext
   SECRET_KEY=your_secret_key
   DEBUG=True
   ALLOWED_HOSTS=localhost,127.0.0.1
   ```

5. **Run migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Start the Django development server**:
   ```bash
   python manage.py runserver
   ```

### API Endpoints

- **POST `/api/analyze_transcription/`**: Analyzes the provided text transcription for potential scams.

#### Example Request

```bash
curl -X POST http://127.0.0.1:8000/api/analyze_transcription/ -d "text=You have won a lottery. Click here to claim your prize."
```

#### Example Response

```json
{
  "result": "spam",
  "transcription": "You have won a lottery. Click here to claim your prize."
}
```

## Frontend Setup

### Overview

The frontend is a PyQt desktop application that captures real-time audio, detects ringing noises, transcribes audio, and interacts with the Django backend to analyze the transcription.

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/scamshield.git
   cd scamshield/frontend
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** in the frontend directory with the following content:
   ```plaintext
   BACKEND_URL=http://127.0.0.1:8000
   ```

5. **Run the PyQt application**:
   ```bash
   python main.py
   ```

### Application Features

- **Real-Time Audio Capture**: Listens for ringing noises to start analysis.
- **Transcription**: Uses Google Speech Recognition to transcribe audio.
- **Scam Detection**: Sends transcription to the backend for scam detection.
- **User Interface**: Displays transcription, detection results, and status updates.

## Running the Application

1. **Start the Django backend server**:
   ```bash
   python manage.py runserver
   ```

2. **Run the PyQt frontend application**:
   ```bash
   python main.py
   ```

ScamShield will start listening for incoming calls, transcribe audio in real-time, and analyze the transcriptions for potential scams, displaying the results in the user interface.

The machine learning model files were too large to upload directly to GitHub, so I compressed the necessary files for easier handling. If you would like to review the complete project, you can access all the files at the following link: https://mega.nz/folder/DPJlnByb#DneGrPOGNK0Ii3BivealhQ

