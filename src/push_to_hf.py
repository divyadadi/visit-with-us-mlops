import os
from huggingface_hub import HfApi

MODEL_REPO = os.getenv("HF_MODEL_REPO", "ddadid/visit-with-us-wellness-tourism")

def main():
    token = os.environ["HF_TOKEN"]
    api = HfApi()

    api.upload_file(
        path_or_fileobj="artifacts/model.joblib",
        path_in_repo="ci_artifacts/model.joblib",
        repo_id=MODEL_REPO,
        repo_type="model",
        token=token
    )

    api.upload_file(
        path_or_fileobj="artifacts/metrics.json",
        path_in_repo="ci_artifacts/metrics.json",
        repo_id=MODEL_REPO,
        repo_type="model",
        token=token
    )

    api.upload_file(
        path_or_fileobj="artifacts/best_params.json",
        path_in_repo="ci_artifacts/best_params.json",
        repo_id=MODEL_REPO,
        repo_type="model",
        token=token
    )

    print("✅ Uploaded CI artifacts to HF Model Hub:", MODEL_REPO)

if __name__ == "__main__":
    main()
