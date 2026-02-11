import praw
import pandas as pd
import datetime
import time
import logging

# --- 1. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    filename='reddit_scraper.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- 2. PRAW API Setup ---
CLIENT_ID = "USER_CLIENT_ID"
CLIENT_SECRET = "USER_CLIENT_SECRET"
USER_AGENT = "AI Discourse Scraper v0.1 by /u/RedditUsername"

# --- 3. Configuration ---
TARGET_SUBREDDITS = [
    "singularity",
    "MachineLearning",
    "artificial",
    "ArtificialInteligence",
    "LocalLLaMA",
    "StableDiffusion",
    "learnmachinelearning",
    "generativeAI",
    "PromptEngineering",
    "deeplearning",
    "ClaudeAI",
    "Futurology",
    "technology",
    "DarkFuturology",
    "compsci",
    "ControlProblem",
    "LocalLLM",
    "OpenAI",
    "Anthropic",
    "reinforcementlearning",
    "AI_Agents",
    "aiengineering",
    "MLQuestions",
    "LanguageTechnology",
    "AgentsOfAI",
    "GoogleGeminiAI",
    "GeminiAI",
    "DeepSeek"
]

# Timeframe: July 23, 2025, to October 31, 2025
START_DATE_UTC = datetime.datetime(2025, 7, 23, tzinfo=datetime.timezone.utc).timestamp()
END_DATE_UTC = datetime.datetime(2025, 10, 31, tzinfo=datetime.timezone.utc).timestamp()

OUTPUT_FILE = "dataset.csv" # Save raw data

# --- 4. Data Collection Function ---
def fetch_reddit_data():
    """
    Fetches recent post data from the target subreddits and saves to CSV.
    """
    # Initialize PRAW
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT
        )
        logging.info("PRAW Reddit instance initialized.")
        print("PRAW Reddit instance initialized.")
    except Exception as e:
        logging.error(f"Error initializing PRAW: {e}")
        print(f"Error initializing PRAW: {e}")
        return

    # Verify authentication
    try:
        reddit.user.me()
        logging.info("Reddit API authentication successful.")
        print("Authentication successful.")
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        print(f"Authentication failed: {e}")
        return

    all_posts = []
    print(f"Fetching posts from {datetime.datetime.fromtimestamp(START_DATE_UTC, tz=datetime.timezone.utc)} to {datetime.datetime.fromtimestamp(END_DATE_UTC, tz=datetime.timezone.utc)}")
    logging.info(f"Fetching posts from {datetime.datetime.fromtimestamp(START_DATE_UTC, tz=datetime.timezone.utc)} to {datetime.datetime.fromtimestamp(END_DATE_UTC, tz=datetime.timezone.utc)}")

    for sub in TARGET_SUBREDDITS:
        print(f"\n--- Processing subreddit: r/{sub} ---")
        logging.info(f"Processing subreddit: r/{sub}")
        
        subreddit = reddit.subreddit(sub)
        after = None
        posts_collected_this_sub = 0
        keep_paginating = True
        
        # We set a failsafe limit of 20 pages (20 * 1000 = 20,000 posts)
        # The loop will *naturally* stop when it hits the START_DATE_UTC.
        for page_num in range(20): 
            if not keep_paginating:
                logging.info(f"Stopping pagination for r/{sub}.")
                break
            
            print(f"Fetching page {page_num + 1} for r/{sub}...")
            posts_this_page = []
            
            try:
                # Fetch the page generator
                post_generator = subreddit.new(limit=1000, params={'after': after} if after else {})
                
                # CRITICAL FIX 1: Consume the generator into a list
                posts_this_page = list(post_generator)
                
                if not posts_this_page:
                    logging.info(f"No more posts returned for r/{sub}. Stopping.")
                    keep_paginating = False
                    break
                    
            except Exception as e:
                logging.error(f"Error fetching page {page_num + 1} for r/{sub}: {e}")
                keep_paginating = False
                break

            # Process the posts from the list we just collected
            for post in posts_this_page:
                if START_DATE_UTC <= post.created_utc <= END_DATE_UTC:
                    all_posts.append({
                        "id": post.id,
                        "subreddit": sub,
                        "title": post.title,
                        "selftext": post.selftext or "",
                        "score": post.score,
                        "created_utc": post.created_utc,
                        "timestamp": datetime.datetime.fromtimestamp(post.created_utc, tz=datetime.timezone.utc)
                    })
                    posts_collected_this_sub += 1
                
                if post.created_utc < START_DATE_UTC:
                    logging.info(f"Reached posts older than start date in r/{sub}.")
                    keep_paginating = False
                    break # Break from inner 'for post' loop
            
            # CRITICAL FIX 2: Set 'after' for the next page
            after = posts_this_page[-1].name 
            
            logging.info(f"Page {page_num + 1} processed. Collected {posts_collected_this_sub} posts so far from r/{sub}.")
            
            # Be polite to the API
            time.sleep(2) 

        print(f"Collected {posts_collected_this_sub} posts from r/{sub} within our timeframe.")
        logging.info(f"Collected {posts_collected_this_sub} posts from r/{sub} within our timeframe.")

    if not all_posts:
        logging.warning("No posts found. Check API credentials, subreddit names, or timeframe.")
        print("No posts found. Check API credentials, subreddit names, or timeframe.")
        return

    # --- 5. Save RAW Data to CSV ---
    print(f"\nTotal posts collected: {len(all_posts)}")
    logging.info(f"Total posts collected: {len(all_posts)}")
    df = pd.DataFrame(all_posts)
    
    # CRITICAL FIX 3: We create the 'document' column but DO NOT clean it.
    # We save the raw data. Preprocessing happens in the analysis script.
    print("Creating 'document' column from raw text...")
    logging.info("Creating 'document' column from raw text...")
    df['document'] = df['title'] + " " + df['selftext']

    # Save to CSV
    try:
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"Successfully saved RAW data to {OUTPUT_FILE}")
        logging.info(f"Successfully saved RAW data to {OUTPUT_FILE}")
    except Exception as e:
        logging.error(f"Error saving to CSV: {e}")
        print(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    fetch_reddit_data()
