import pandas as pd
from transformers import pipeline
from scipy.stats import mannwhitneyu
from tqdm import tqdm

# --- 1. Load Data ---
df_agents = pd.read_csv("topic_2_posts.csv") # Builders
df_jobs = pd.read_csv("topic_5_posts.csv")   # Workers

docs_agents = df_agents['document'].dropna().astype(str).tolist()
docs_jobs = df_jobs['document'].dropna().astype(str).tolist()

# --- 2. Initialize Standard Sentiment Model (The Baseline) ---
# This model only detects POSITIVE vs NEGATIVE (no "Anxiety" or "Confusion")
print("Loading baseline sentiment model...")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True, max_length=512)

# --- 3. Helper Function ---
def get_negative_scores(texts):
    scores = []
    print(f"Processing {len(texts)} documents...")
    for i in tqdm(range(0, len(texts), 16)):
        batch = texts[i:i+16]
        results = sentiment_pipeline(batch)
        for res in results:
            # If label is NEGATIVE, take the score. If POSITIVE, score is 1 - score (or 0 for strict mapping).
            # Let's just track "Negative Probability"
            if res['label'] == 'NEGATIVE':
                scores.append(res['score'])
            else:
                scores.append(1.0 - res['score']) # Low negative score
    return scores

# --- 4. Run Inference ---
print("Calculating Baseline Sentiment for Builders...")
builders_neg = get_negative_scores(docs_agents)

print("Calculating Baseline Sentiment for Workers...")
workers_neg = get_negative_scores(docs_jobs)

# --- 5. Statistical Test ---
stat, p_value = mannwhitneyu(builders_neg, workers_neg, alternative='two-sided')

print("\n=== ABLATION RESULTS (Standard Sentiment) ===")
print(f"Builders Mean 'Negative' Score: {sum(builders_neg)/len(builders_neg):.4f}")
print(f"Workers Mean 'Negative' Score:  {sum(workers_neg)/len(workers_neg):.4f}")
print(f"P-Value: {p_value}")
