import pandas as pd
from transformers import pipeline
from scipy.stats import mannwhitneyu
from tqdm import tqdm
import re

# --- 1. Load Data ---
df_jobs = pd.read_csv("topic_5_posts.csv") # Workers only (we are testing robustness here)
docs_jobs = df_jobs['document'].dropna().astype(str).tolist()

# --- 2. Define the Mask ---
# These are the top words from Topic 5 Word Cloud
mask_terms = [
    "job", "jobs", "career", "careers", "interview", "interviews",
    "resume", "cv", "hiring", "hired", "offer", "salary",
    "internship", "degree", "market", "work", "company"
]

def mask_text(text_list, terms):
    masked_docs = []
    # Regex to remove whole words, case insensitive
    pattern = re.compile(r'\b(' + '|'.join(terms) + r')\b', re.IGNORECASE)

    for doc in text_list:
        # Replace keywords with nothing
        cleaned = pattern.sub('', doc)
        # Clean up double spaces
        cleaned = re.sub(' +', ' ', cleaned).strip()
        masked_docs.append(cleaned)
    return masked_docs

print("Masking topic-specific vocabulary...")
docs_jobs_masked = mask_text(docs_jobs, mask_terms)

# Print a sample to check
print(f"Original: {docs_jobs[0][:100]}...")
print(f"Masked:   {docs_jobs_masked[0][:100]}...")

# --- 3. Initialize Model ---
print("\nLoading Emotion Model...")
emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    truncation=True,
    max_length=512
)

# --- 4. Re-Calculate Scores (Masked) ---
def get_anxiety_scores(texts):
    scores = []
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        results = emotion_classifier(batch)
        for res in results:
            total = sum(item['score'] for item in res if item['label'] in ['fear', 'nervousness'])
            scores.append(total)
    return scores

print("Calculating Anxiety on MASKED text...")
masked_anxiety = get_anxiety_scores(docs_jobs_masked)

# --- 5. Compare to Original Baseline (Approximate) ---
# We compare the Masked Mean to the Original Mean
original_mean = 0.0116
masked_mean = sum(masked_anxiety) / len(masked_anxiety)

print("\n=== ABLATION 3 RESULTS (Lexical Masking) ===")
print(f"Original Worker Anxiety Mean: {original_mean}")
print(f"Masked Worker Anxiety Mean:   {masked_mean:.4f}")
print(f"Retention Rate: {(masked_mean / original_mean) * 100:.1f}%")
