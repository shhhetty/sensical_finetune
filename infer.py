import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize client
client = OpenAI()

# Load test keywords
df = pd.read_csv("test.csv")

FINETUNED_MODEL = "ft:gpt-4.1-mini-2025-04-14:****"

def get_prediction(kw):
    try:
        resp = client.chat.completions.create(
            model=FINETUNED_MODEL,
            messages=[
                {"role": "system", "content": "You are a Quality Assurance Analyst for Lenovo.com. Your goal is to determine if a keyword can serve as the title of a landing page showing a grid of products. It must be a suitable H1 heading for a category page. Respond with '1' if it is a sensible, shoppable category. Respond with '0' if it is not."},
                {"role": "user", "content": kw}
            ],
            temperature=0
        )
        output = resp.choices[0].message.content.strip()
        return {"keyword": kw, "prediction": output}
    except Exception as e:
        return {"keyword": kw, "prediction": "error", "error": str(e)}

results = []
max_workers = 50 # Adjust based on your rate limit and system

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(get_prediction, kw) for kw in df["keyword"]]
    for future in as_completed(futures):
        results.append(future.result())

results_df = pd.DataFrame(results)
# Merge with input to preserve order
ordered_df = pd.merge(df, results_df, on="keyword", how="left")
ordered_df.to_csv("lnv_output.csv", index=False)

print("✅ Predictions saved to lnv_output.csv")
