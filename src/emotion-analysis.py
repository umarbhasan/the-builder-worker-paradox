import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from transformers import pipeline
from tqdm import tqdm
import os
from wordcloud import WordCloud

# --- 1. Load Data from CSVs ---
file_agents = "topic_2_posts.csv" # Update if needed
file_jobs = "topic_5_posts.csv"   # Update if needed

if not os.path.exists(file_agents) or not os.path.exists(file_jobs):
    print("CRITICAL ERROR: Topic CSV files not found.")
else:
    print(f"Loading data from {file_agents} and {file_jobs}...")
    df_agents = pd.read_csv(file_agents)
    df_jobs = pd.read_csv(file_jobs)

    docs_agents = df_agents['document'].dropna().astype(str).tolist()
    docs_jobs = df_jobs['document'].dropna().astype(str).tolist()

    # --- 2. Setup SOTA Emotion Classifier ---
    # We use a model trained on GoEmotions (Reddit data) with 28 labels.
    # This captures 'Curiosity', 'Confusion', 'Nervousness' which standard models miss.
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading GoEmotions Classifier on device: {device}...")

    emotion_classifier = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None, # Return scores for ALL 28 labels
        device=device,
        truncation=True
    )

    # --- 3. Run Inference ---
    def get_average_emotions(text_list, batch_size=32):
        # We track specific emotions relevant to the "Builder-Worker Paradox"
        # We merge synonyms to make the chart readable.
        target_metrics = {
            'Anxiety': 0,      # fear + nervousness
            'Curiosity': 0,    # curiosity
            'Confusion': 0,    # confusion
            'Neutral': 0,      # neutral
            'Optimism': 0,     # optimism + approval
            'Sadness': 0       # sadness + disappointment
        }
        count = 0

        for i in tqdm(range(0, len(text_list), batch_size)):
            batch = text_list[i:i+batch_size]
            try:
                results = emotion_classifier(batch)
                for res in results:
                    # res is a list of dicts [{'label': 'joy', 'score': 0.9}, ...]
                    scores = {item['label']: item['score'] for item in res}

                    # Aggregate into our hypothesis buckets
                    target_metrics['Anxiety'] += scores.get('fear', 0) + scores.get('nervousness', 0)
                    target_metrics['Curiosity'] += scores.get('curiosity', 0)
                    target_metrics['Confusion'] += scores.get('confusion', 0)
                    target_metrics['Neutral'] += scores.get('neutral', 0)
                    target_metrics['Optimism'] += scores.get('optimism', 0) + scores.get('approval', 0)
                    target_metrics['Sadness'] += scores.get('sadness', 0) + scores.get('disappointment', 0)

                    count += 1
            except Exception as e:
                continue

        if count == 0: return target_metrics
        return {k: v / count for k, v in target_metrics.items()}

    print("Processing Agents Topic (Builders)...")
    emotions_agents = get_average_emotions(docs_agents)
    print("Processing Jobs Topic (Workers)...")
    emotions_jobs = get_average_emotions(docs_jobs)

    # --- 4. Generate Radar Chart ---
    def plot_radar_chart(emotions_a, emotions_b, label_a, label_b, title, filename):
        categories = list(emotions_a.keys())
        N = len(categories)
        values_a = list(emotions_a.values())
        values_b = list(emotions_b.values())

        # Close the loop
        values_a += values_a[:1]
        values_b += values_b[:1]
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        plt.xticks(angles[:-1], categories, color='grey', size=12)
        ax.set_rlabel_position(0)
        plt.yticks([0.1, 0.2, 0.3, 0.4], ["0.1", "0.2", "0.3", "0.4"], color="grey", size=10)
        plt.ylim(0, 0.5)

        ax.plot(angles, values_a, linewidth=2, linestyle='solid', label=label_a, color='blue')
        ax.fill(angles, values_a, 'blue', alpha=0.1)

        ax.plot(angles, values_b, linewidth=2, linestyle='solid', label=label_b, color='red')
        ax.fill(angles, values_b, 'red', alpha=0.1)

        plt.title(title, size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.show()

    plot_radar_chart(
        emotions_agents,
        emotions_jobs,
        "Builders (Agents)",
        "Workers (Jobs)",
        "Emotional Profile: Builders vs. Workers",
        "emotion_radar_chart.pdf"
    )
    print("Radar chart saved.")

    # --- 5. Generate Word Clouds (Optional) ---
    def generate_wordcloud(text_list, title, filename):
        text = " ".join(text_list)
        wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.show()

    print("Generating Word Clouds (for Appendix)...")
    generate_wordcloud(docs_agents, "Topic 2: Agents (Word Cloud)", "wordcloud_agents.pdf")
    generate_wordcloud(docs_jobs, "Topic 5: Jobs (Word Cloud)", "wordcloud_jobs.pdf")
