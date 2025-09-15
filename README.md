# Sensical Finetune

This repository contains scripts and data for fine-tuning a machine learning model.

## Project Overview

This project is designed to fine-tune a model for a specific task. The general workflow involves:

1.  **Data Preparation**: Raw data is processed and formatted for training.
2.  **Model Fine-Tuning**: A pre-trained model is fine-tuned on the prepared dataset.
3.  **Inference**: The fine-tuned model is used to make predictions on new data.

## Files

*   **`prepare_data.py`**: This script prepares the data for training and validation. It likely takes a raw dataset as input and outputs `train.jsonl` and `val.jsonl`.
*   **`infer.py`**: This script loads the fine-tuned model and performs inference on new data.
*   **`keywords.csv`**: This file contains keywords that are likely used in the data preparation process.
*   **`test.csv`**: This file contains data for testing the performance of the fine-tuned model.
*   **`train.jsonl`**: The training dataset in JSONL format.
*   **`val.jsonl`**: The validation dataset in JSONL format.
*   **`finetuneflow.docx`**: This document likely provides a more detailed overview of the project's workflow.

## Usage

### 1. Prepare the Data

Run the `prepare_data.py` script to process your raw data and create the training and validation sets.

```bash
python prepare_data.py
```

### 2. Fine-Tune the Model

*(Instructions on how to run the fine-tuning process would go here. This information is not available in the provided file list.)*

### 3. Run Inference

Use the `infer.py` script to make predictions with your fine-tuned model.

```bash
python infer.py
```

## Data Format

The training and validation data are in the JSONL format, where each line is a JSON object. The `test.csv` file is a comma-separated values file.
