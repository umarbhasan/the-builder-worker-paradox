import pandas as pd
from scipy.stats import mannwhitneyu
import re

# --- 1. Load Data ---
df_agents = pd.read_csv("topic_2_posts.csv")
df_jobs = pd.read_csv("topic_5_posts.csv")

docs_agents = df_agents['document'].dropna().astype(str).tolist()
docs_jobs = df_jobs['document'].dropna().astype(str).tolist()

# --- 2. Define Naive Keywords ---
# A standard list of anxiety-related terms
keywords = ['anxiety', 'anxious', 'scared', 'afraid', 'worry', 'worried', 'nervous', 'panic', 'doom', 'fear', 'terrified']

def calculate_keyword_density(texts, keywords):
    scores = []
    for text in texts:
        text_lower = text.lower()
        # Count total occurrences of keywords in the post
        count = sum(1 for word in keywords if word in text_lower)
        scores.append(count)
    return scores

# --- 3. Run Inference ---
builders_kw = calculate_keyword_density(docs_agents, keywords)
workers_kw = calculate_keyword_density(docs_jobs, keywords)

# --- 4. Statistical Test ---
stat, p_value = mannwhitneyu(builders_kw, workers_kw, alternative='two-sided')

print("\n=== ABLATION 2 RESULTS (Naive Keyword Search) ===")
print(f"Builders Mean Keyword Count: {sum(builders_kw)/len(builders_kw):.4f}")
print(f"Workers Mean Keyword Count:  {sum(workers_kw)/len(workers_kw):.4f}")
print(f"P-Value: {p_value}")
