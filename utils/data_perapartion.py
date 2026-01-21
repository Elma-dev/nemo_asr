from huggingface_hub import login
from dotenv import load_dotenv
import os
from datasets import load_dataset
import json
import soundfile as sf


load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)


def load_export_dataset(
    dataset_name: str,
    name: str = "",
    dataset_split: str = "train",
    text_column: str = "text",
    audio_column: str = "audio",
    folder_split: str = "train",
    out_manifest: str = "train.json",
):
    ds = load_dataset(dataset_name, name=name, split=dataset_split)
    os.makedirs(f"data/{folder_split}", exist_ok=True)
    with open(out_manifest, "w") as f:
        for i, item in enumerate(ds):
            audio_path = f"data/{folder_split}/audio_{i}.wav"
            sf.write(audio_path, item[audio_column]["array"], 16000)

            entry = {
                "audio_filepath": os.path.abspath(audio_path),
                "duration": len(item[audio_column]["array"]) / 16000,
                "text": item[text_column],
            }
            f.write(json.dumps(entry) + "\n")
