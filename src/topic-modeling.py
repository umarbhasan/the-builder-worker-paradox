# BERTopic Script
# This script loads 'dataset.csv', performs preprocessing non-destructively,
# includes Chrome installation for Kaleido PDF exports, tunes BERTopic to reduce outliers,
# and extracts raw posts for the top 12 topics.

import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import re
import nltk
from nltk.corpus import stopwords
import os
from hdbscan import HDBSCAN
from umap import UMAP

# --- 0. Install Dependencies for Kaleido PDF Exports ---
print("Upgrading kaleido and plotly...")
!pip install -U kaleido plotly

print("Installing Google Chrome and dependencies for Kaleido PDF exports...")
!apt-get update
!apt-get install -y wget unzip libxss1 libappindicator3-1 libasound2 libatk1.0-0 libc6 libcairo2 \
    libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgbm1 libgcc1 libglib2.0-0 libgtk-3-0 \
    libnspr4 libpango-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 \
    libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 \
    libxtst6
!wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
!dpkg -i google-chrome-stable_current_amd64.deb
!apt-get -f install -y
print("Chrome installation complete.")

# --- 1. Data Loading ---
DATA_FILE = "/content/dataset.csv"

if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found.")
    print("Please run 'reddit_data_collector.py' first to generate the data.")
    exit()

print(f"Loading raw dataset from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

df = df.dropna(subset=['document'])
df = df[df['document'].str.strip() != '']

timestamps = pd.to_datetime(df['created_utc'], unit='s')
docs = df['document'].tolist()
print(f"Loaded {len(docs)} raw documents.")

# --- 2. Preprocessing (Non-Destructive) ---
try:
    stop_words_set = set(stopwords.words('english'))
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    stop_words_set = set(stopwords.words('english'))

custom_stopwords = {'ai', 'reddit', 'post', 'comment', 'http', 'https', 'www', 'com', 'org', 'r', 'like', 'get', 'one', 'would', 'people'}  # Removed "data", "model", "use"
stop_words_set.update(custom_stopwords)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|[\n\r]+|[^\w\s]', ' ', text)
    text = ' '.join(word for word in text.split() if word not in stop_words_set and len(word) > 2)
    return text.strip()

print("Preprocessing documents in memory...")
preprocessed_docs = [clean_text(doc) for doc in docs]
print("Preprocessing complete.")

# --- 3. BERTopic Model Configuration ---
print("Initializing embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Initializing BERTopic model with tuning...")
hdbscan_model = HDBSCAN(min_cluster_size=30, min_samples=2)  # Adjusted for less strict clustering
topic_model = BERTopic(
    embedding_model=embedding_model,
    hdbscan_model=hdbscan_model,
    min_topic_size=30,  # Middle ground to balance topics and outliers
    verbose=True
)

# --- 4. Model Training (Phase 3) ---
print("Training BERTopic model... This may take a while (use GPU in Kaggle/Colab).")
topics, probabilities = topic_model.fit_transform(preprocessed_docs)

# --- 5. Exploring the Results ---
print("\n--- Model Training Complete ---")
print(f"BERTopic found {len(topic_model.get_topic_info()) - 1} topics (plus 1 outlier topic).")
print("\nMost frequent topics:")
print(topic_model.get_topic_info().head(12))

# --- 6. Visualization (Answering RQ1) ---
print("\nGenerating and saving visualizations...")

try:
    fig_intertopic = topic_model.visualize_topics()
    fig_intertopic.write_html("intertopic_distance_map.html")
    try:
        fig_intertopic.write_image("intertopic_distance_map.pdf")
        print("Saved intertopic_distance_map.html / .pdf")
    except Exception as e:
        print(f"Failed to save intertopic_distance_map.pdf: {e}")
        print("Saved intertopic_distance_map.html only")

    fig_barchart = topic_model.visualize_barchart(top_n_topics=12)
    fig_barchart.write_html("topic_barcharts.html")
    try:
        fig_barchart.write_image("topic_barcharts.pdf")
        print("Saved topic_barcharts.html / .pdf")
    except Exception as e:
        print(f"Failed to save topic_barcharts.pdf: {e}")
        print("Saved topic_barcharts.html only")

    print("Generating 'topics_over_time.html'...")
    try:
        topics_over_time = topic_model.topics_over_time(preprocessed_docs, timestamps)
        fig_over_time = topic_model.visualize_topics_over_time(topics_over_time)
        fig_over_time.write_html("topics_over_time.html")
        try:
            fig_over_time.write_image("topics_over_time.pdf")
            print("Saved topics_over_time.html / .pdf")
        except Exception as e:
            print(f"Failed to save topics_over_time.pdf: {e}")
            print("Saved topics_over_time.html only")
    except Exception as e:
        print(f"Could not generate topics over time visualization: {e}")
        print("This can happen if the dataset is too small or timestamps are invalid.")

except Exception as e:
    print(f"Visualization failed: {e}")
    print("Check if Chrome is installed correctly or skip PDF exports.")

# --- 7. Phase 4: Qualitative Data Extraction ---
print("\nExtracting raw posts for the top 12 topics...")
# Get topic assignments and original documents
topic_info = topic_model.get_topic_info()
top_12_topic_ids = topic_info.sort_values('Count', ascending=False).head(12).index.tolist()

for topic_id in top_12_topic_ids:
    topic_docs = [doc for tid, doc in zip(topics, docs) if tid == topic_id]
    topic_df = pd.DataFrame({'document': topic_docs})
    topic_filename = f"topic_{topic_id}_posts.csv"
    topic_df.to_csv(topic_filename, index=False)
    print(f"Saved {len(topic_docs)} posts to {topic_filename}")
