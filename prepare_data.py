import pandas as pd
from sklearn.model_selection import train_test_split
import json

# --- CONFIGURATION ---
INPUT_CSV_FILE = "/Users/satvikshetty/Downloads/for_classifier - Anntaylor (from tot20 summer qa to T19).csv"
TRAIN_JSONL_FILE = "train_ann.jsonl"
VAL_JSONL_FILE = "val_ann.jsonl"
SYSTEM_PROMPT = (
    "You are a Quality Assurance Analyst for Anntaylor.com. Your goal is to determine if a "
    "keyword can serve as the title of a landing page showing a grid of products. It must be a suitable H1 heading for a category page. "
    "Respond with '1' if it is a sensible, shoppable category. Respond with '0' if it is not."
)

def create_jsonl_file(df, filename):
    """Converts a DataFrame to a JSONL file in the required OpenAI chat format."""
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            # Create the JSON object for each row
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["Keyword"]},
                    {"role": "assistant", "content": str(row["binary_label"])}
                ]
            }
            # Write the JSON object as a new line in the file
            f.write(json.dumps(record) + "\n")

def main():
    """Main function to process the data."""
    print("Starting data preparation...")
    
    # Step 1: Load the dataset
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
        print(f"Successfully loaded {INPUT_CSV_FILE} with {len(df)} rows.")
    except FileNotFoundError:
        print(f"ERROR: The file '{INPUT_CSV_FILE}' was not found.")
        print("Please make sure your CSV file is in the same folder and named correctly.")
        return

    # Step 2: Normalize the labels into a new 'binary_label' column
    # Rule: Keep 1 as 1, everything else (0, -1, 0.5, 2) becomes 0.
    df["binary_label"] = df["Sensical QA"].apply(lambda x: 1 if x == 1 else 0)
    print("Normalized labels to binary (0 or 1).")
    print(df["binary_label"].value_counts())

    # Step 3: Split the data into training and validation sets (80/20)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,       # 20% for validation
        random_state=42,     # Ensures the split is the same every time
        stratify=df["binary_label"] # Ensures train and val sets have similar class distribution
    )
    print(f"Data split into {len(train_df)} training samples and {len(val_df)} validation samples.")

    # Step 4: Convert the DataFrames to JSONL format
    create_jsonl_file(train_df, TRAIN_JSONL_FILE)
    print(f"Successfully created training file: {TRAIN_JSONL_FILE}")
    
    create_jsonl_file(val_df, VAL_JSONL_FILE)
    print(f"Successfully created validation file: {VAL_JSONL_FILE}")
    
    print("\nData preparation complete!")

if __name__ == "__main__":
    main()
