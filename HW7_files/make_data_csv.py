# make_data_csv.py
# Creates a tiny CSV (data.csv) from WikiText-2 (raw) for fast CPU training.
# Produces a single-column CSV with header 'text' and a few thousand short lines.

from datasets import load_dataset
import pandas as pd

def main():
    # Load the raw WikiText-2 train split (small, clean English paragraphs)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    # Keep non-empty lines and trim whitespace to reduce noise
    texts = [x["text"].strip() for x in ds if x["text"] and len(x["text"].strip()) > 0]

    # Subsample to a tiny size for CPU (adjust count as needed for runtime)
    texts = texts[:3000]  # ≈ a few MB on disk, fast to train

    # Save a single-column CSV with header 'text'
    pd.DataFrame({"text": texts}).to_csv("data.csv", index=False)

if __name__ == "__main__":
    main()
