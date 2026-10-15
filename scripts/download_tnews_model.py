"""Download a pinned, public Qwen checkpoint into this project."""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", str(Path("data/hf_cache").resolve()))
from huggingface_hub import HfApi, snapshot_download

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
revision = HfApi().model_info(model_id).sha
destination = Path("models/Qwen2.5-1.5B-Instruct")
print(f"Downloading {model_id} at {revision}", flush=True)
for attempt in range(5):
    try:
        snapshot_download(
            model_id, revision=revision, local_dir=destination, max_workers=3,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab.json", "merges.txt", "README.md", "LICENSE"],
        )
        break
    except Exception as error:
        if attempt == 4:
            raise
        print(f"Download interrupted ({type(error).__name__}); resume attempt {attempt + 2}", flush=True)
        time.sleep(5)
(destination / "download_manifest.json").write_text(
    json.dumps({"model_id": model_id, "revision": revision}, indent=2) + "\n"
)
print(f"Saved model to {destination}", flush=True)
