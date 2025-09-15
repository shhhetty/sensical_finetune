# Sensical Finetune

This repository contains the scripts and data for fine-tuning a machine learning model to generate sensical responses.

## Project Overview

This project is designed to fine-tune a large language model (LLM) to generate more sensical and relevant text based on a given input. The workflow involves these primary stages:

1.  **Data Preparation**: Raw data is processed, cleaned, and formatted into training and validation sets.
2.  **Model Fine-Tuning**: A pre-trained LLM is fine-tuned using the prepared datasets.
3.  **Inference**: The fine-tuned model is used to generate predictions on new, unseen data.

## Files

Here is a breakdown of the files in this repository and their roles:

*   **`prepare_data.py`**: This script prepares the data for training and validation. It takes raw data as input and generates `train.jsonl` and `val.jsonl` files.
*   **`infer.py`**: This script loads the fine-tuned model and performs inference on new data.
*   **`keywords.csv`**: This file contains a list of keywords that are used in the data preparation process, likely for filtering or categorization.
*   **`test.csv`**: This file contains the test data used to evaluate the performance of the fine-tuned model.
*   **`train.jsonl`**: The training dataset in JSONL format. Each line represents a single training example.
*   **`val.jsonl`**: The validation dataset in JSONL format, used for monitoring the model's performance during training.
*   **`finetuneflow.docx`**: This document provides a detailed overview of the project's workflow, architecture, and fine-tuning process.

## Usage

To use this project, follow these steps:

### 1. Prepare the Data

Run the `prepare_data.py` script to process your raw data and create the training and validation sets.

```shell
python prepare_data.py
```

### 2. Fine-Tune the Model

*(Instructions on how to run the fine-tuning process would go here. Please refer to the `finetuneflow.docx` document for detailed instructions.)*

### 3. Run Inference

Use the `infer.py` script to make predictions with your fine-tuned model.

```shell
python infer.py
```

## Data Format

The training and validation data are in the JSONL format, where each line is a JSON object representing a single data point. The `test.csv` file is a comma-separated values (CSV) file.

## Contributing

Contributions to this project are welcome! Please feel free to fork the repository, make your changes, and submit a pull request.
