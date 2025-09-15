import pandas as pd
from openai import OpenAI

# Initialize client
client = OpenAI()

# Load test keywords
df = pd.read_csv("test.csv")

FINETUNED_MODEL = "ft:gpt-4.1-mini-2025-04-14:***"

results = []

for kw in df["keyword"]:
    resp = client.chat.completions.create(
        model=FINETUNED_MODEL,
        messages=[
            {"role": "system", "content": "You are a Quality Assurance Analyst for Dillards.com. Your goal is to determine if a keyword can serve as the title of a landing page showing a grid of products. It must be a suitable H1 heading for a category page. Respond with '1' if it is a sensible, shoppable category. Respond with '0' if it is not."},
            {"role": "user", "content": kw}
        ],
        temperature=0  # deterministic
    )
    output = resp.choices[0].message.content.strip()
    results.append({"keyword": kw, "prediction": output})

# Save results to CSV
out_df = pd.DataFrame(results)
out_df.to_csv("predictionss.csv", index=False)

print("✅ Predictions saved to predictions.csv")
