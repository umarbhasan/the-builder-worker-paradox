import pandas as pd
from transformers import pipeline
from scipy.stats import mannwhitneyu
from tqdm import tqdm  # specific library for progress bars

# --- 1. Load the Data ---
# We load the files of interest
df_agents = pd.read_csv("topic_2_posts.csv") # Builders
df_jobs = pd.read_csv("topic_5_posts.csv")   # Workers

# Clean the data (remove empty rows)
docs_agents = df_agents['document'].dropna().astype(str).tolist()
docs_jobs = df_jobs['document'].dropna().astype(str).tolist()

# --- 2. Initialize the Emotion Model ---
# We use the GoEmotions model which can detect 28 different emotions
print("Loading model... (this may take a moment)")
emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,  # Return scores for ALL labels, not just the top one
    truncation=True,
    max_length=512
)

# --- 3. Define Helper Function to Extract Scores ---
def extract_category_scores(texts, category_labels):
    """
    Runs the classifier on a list of texts and sums up scores for specific labels.
    e.g., category_labels=['fear', 'nervousness'] -> returns a list of Anxiety scores.
    """
    scores = []
    print(f"Processing {len(texts)} documents...")

    # Process in batches of 16 for speed
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        results = emotion_classifier(batch)

        for res in results:
            # 'res' is a list of dicts: [{'label': 'fear', 'score': 0.9}, {'label': 'joy', 'score': 0.01}...]
            # Sum the scores for the labels we care about (e.g., fear + nervousness)
            total_score = sum(
                item['score'] for item in res if item['label'] in category_labels
            )
            scores.append(total_score)

    return scores

# --- 4. Calculate Scores for Each Group ---
# We define "Anxiety" as the sum of 'fear' and 'nervousness'
anxiety_labels = ['fear', 'nervousness']

print("\n--- Calculating Anxiety Scores for Builders (Topic 2) ---")
builders_anxiety = extract_category_scores(docs_agents, anxiety_labels)

print("\n--- Calculating Anxiety Scores for Workers (Topic 5) ---")
workers_anxiety = extract_category_scores(docs_jobs, anxiety_labels)

# --- 5. Perform the Statistical Test (Mann-Whitney U) ---
# We use Mann-Whitney because emotion scores are not normally distributed (bell curve).
stat, p_value = mannwhitneyu(builders_anxiety, workers_anxiety, alternative='two-sided')

print("\n================RESULTS================")
print(f"Builders (Topic 2) Mean Anxiety Score: {sum(builders_anxiety)/len(builders_anxiety):.4f}")
print(f"Workers (Topic 5) Mean Anxiety Score:  {sum(workers_anxiety)/len(workers_anxiety):.4f}")
print(f"Mann-Whitney U Statistic: {stat}")
print(f"P-Value: {p_value}")

if p_value < 0.001:
    print("Result: Statistically Significant (p < 0.001) ***")
elif p_value < 0.05:
    print("Result: Statistically Significant (p < 0.05) *")
else:
    print("Result: Not Significant")
