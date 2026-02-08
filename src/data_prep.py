import os
import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_REPO = os.getenv("HF_DATASET_REPO", "ddadid/visit-with-us-wellness-tourism")
TRAIN_FILE = os.getenv("TRAIN_PATH_IN_DATASET", "processed/train.csv")
TEST_FILE  = os.getenv("TEST_PATH_IN_DATASET", "processed/test.csv")

def main():
    os.makedirs("data", exist_ok=True)

    train_path = hf_hub_download(repo_id=DATASET_REPO, filename=TRAIN_FILE, repo_type="dataset")
    test_path  = hf_hub_download(repo_id=DATASET_REPO, filename=TEST_FILE, repo_type="dataset")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("✅ Downloaded train/test from HF Dataset and saved locally to /data")

if __name__ == "__main__":
    main()
