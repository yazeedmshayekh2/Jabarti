"""
Download, Filter, and Clean the Corpus
======================================
"""

from pathlib import Path

from huggingface_hub import hf_hub_download
import pandas as pd

from jabarti.tokenizer_building.cleaning import (
    clean_dataframe,
)

REPO = "bakrianoo/jabarti-llm-dataset"
NUM_OF_CHUNKS = 20

files = {
    "pretrain_phase1train": "pretrain/phase1_train-00000-of-00001.parquet",
    "pretrain_phase1eval" : "pretrain/phase1_eval-00000-of-00001.parquet",
    "pretrain_phase2train": "pretrain/phase2_train-00000-of-00001.parquet",
    "pretrain_phase2eval" : "pretrain/phase2_eval-00000-of-00001.parquet",

    "tf_train": "finetune/train-00000-of-00001.parquet",
    "tf_eval" : "finetune/eval-00000-of-00001.parquet",
}

TRAIN_SPLITS = ['pretrain_phase1train', 'pretrain_phase1eval']
VAL_SPLITS = ['pretrain_phase2train', 'pretrain_phase2eval']

CURRENT_DIR = Path(__file__).parent
OUTPUT_DIR = CURRENT_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_and_filter():

    for name, filename in files.items():

        print(f"Load: {name}")

        local  = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=filename,
        )
        
        df = pd.read_parquet(local)
        
        if name in TRAIN_SPLITS:

            before_filter = len(df)

            df = (
                df.sort_values(["article_id", "chunk_index"])
                .groupby("article_id", sort=False)
                .head(NUM_OF_CHUNKS)
                .reset_index(drop=True)
            )

            after_filter = len(df)

            print(f"df: {name} => {before_filter} filtered to be {after_filter}")

        output_path = OUTPUT_DIR / f"{name}_filtered.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved: {output_path}")

def clean_phase(phase, input_path, output_path):

    print(f"==== clean_phase: {phase} ====")

    df = pd.read_parquet(input_path)

    print(f"Source      :   {input_path}")
    print(f"Rows        :   {len(df):,}")

    cleaned_df = clean_dataframe(df)

    removed_rows = len(df) - len(cleaned_df)

    print(f"Rows Kept   : {len(cleaned_df):,}  (dropped {removed_rows:,} empty rows)")

    cleaned_df.to_parquet(output_path, index=False)

def main():
    download_and_filter()

    print("STEP 1: LOAD CLEANED CORPUS")

    for name in TRAIN_SPLITS:
        input_path  = OUTPUT_DIR / f"{name}_filtered.parquet"
        output_path = OUTPUT_DIR / f"{name}_cleaned.parquet"

        clean_phase(phase=name, input_path=input_path, output_path=output_path)

    for name in VAL_SPLITS:
        input_path  = OUTPUT_DIR / f"{name}_filtered.parquet"
        output_path = OUTPUT_DIR / f"{name}_cleaned.parquet"

        clean_phase(phase=name, input_path=input_path, output_path=output_path)

if __name__ == "__main__":
    main()