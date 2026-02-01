import requests
import time
import json
import os

# --- CONFIGURATION ---
OUTPUT_FILE = "raw_reddit_data.jsonl"
TARGET_SUBREDDITS = [
    "Ozempic", "Wegovy", "Mounjaro", "Zepbound", "Trulicity", 
    "Victoza", "Rybelsus", "Jardiance", "Farxiga", "Januvia", 
    "Eliquis", "Xarelto", "Entresto", "Humira", "Keytruda", 
    "Opdivo", "Enbrel", "Stelara", "Biktarvy", "Dupixent"
]

LIMIT_PER_DRUG = 1000 
BASE_URL = "https://api.pullpush.io/reddit/search/comment/"

def fetch_batch(subreddit, before_timestamp=None):
    params = {
        "subreddit": subreddit,
        "size": 100,
        "sort": "desc",
        "sort_type": "created_utc"
    }
    if before_timestamp:
        params["before"] = before_timestamp

    for attempt in range(3): # Try 3 times
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json().get('data', [])
                return data
            elif response.status_code == 429:
                time.sleep(10)
            else:
                time.sleep(2)
        except:
            time.sleep(2)
    return []

def run_ingestion():
    print(f"🚀 Starting Smart Ingestion...")
    
    # 1. Load existing IDs so we don't save duplicates
    seen_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen_ids.add(rec['id'])
                except: pass
    print(f"   ℹ️ Resuming... {len(seen_ids)} comments already in database.\n")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for drug in TARGET_SUBREDDITS:
            print(f"🔍 Scraping r/{drug}...")
            count = 0
            before = None 
            consecutive_duplicates = 0
            
            while count < LIMIT_PER_DRUG:
                batch = fetch_batch(drug, before)
                
                if not batch:
                    print(f"      ✅ Reached end of r/{drug} (or API empty).")
                    break
                
                new_items = 0
                for comment in batch:
                    cid = comment.get('id')
                    
                    # STOP if we see duplicates (means we are looping or caught up)
                    if cid in seen_ids:
                        consecutive_duplicates += 1
                        continue
                    
                    consecutive_duplicates = 0 # Reset if we found a new one
                    seen_ids.add(cid)
                    
                    text = comment.get('body', '')
                    if text in ["[deleted]", "[removed]"] or len(text) < 10:
                        continue

                    record = {
                        "text": text,
                        "subreddit": drug, # Force the tag to match the target drug
                        "id": cid,
                        "created_utc": comment.get('created_utc')
                    }
                    f.write(json.dumps(record) + "\n")
                    new_items += 1
                    count += 1
                
                # SAFETY: If we fetched a batch but found 0 new items, we are stuck.
                if new_items == 0:
                    print(f"      ⚠️ No new items found in batch. Stopping r/{drug}.")
                    break

                # Pagination Logic
                last_timestamp = batch[-1].get('created_utc')
                
                # SAFETY: If time didn't move backwards, we are stuck.
                if before and last_timestamp >= before:
                    print("      ⚠️ Time stuck. Stopping.")
                    break
                    
                before = last_timestamp
                print(f"      Collected {count}/{LIMIT_PER_DRUG}...")
                time.sleep(1) # Be polite

            print(f"   🎉 Finished r/{drug}. Total: {count}\n")

if __name__ == "__main__":
    run_ingestion()