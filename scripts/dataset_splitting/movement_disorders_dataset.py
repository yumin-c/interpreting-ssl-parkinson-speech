import os
import re
import glob
import argparse
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Movement Disorders (Kaggle) dataset into a CSV for GITA pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples-dir", required=True, type=str,
                        help="Directory containing restructured .wav files (norm_audios)")
    parser.add_argument("--metadata-path", required=True, type=str,
                        help="Path to metadata.csv produced by movement_disorders_restructure.py")
    parser.add_argument("--output-dir", default="./splits/movement_disorders/", type=str)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_path)
    # label and group_id are stored explicitly in the metadata CSV (supports 3 classes)
    metadata["sex"] = metadata["SEX"].map(lambda x: 1 if x == "M" else 0)

    dataset = []
    wavs = glob.glob(os.path.join(args.samples_dir, "*.wav"), recursive=True)
    for wav_path in tqdm(wavs):
        sample_id = os.path.basename(wav_path).split(".")[0]
        m = re.search(r"(AVPEPUDEA(?:C|AD)?\d{4})", sample_id)
        if not m:
            continue
        subject_id = m.group(1)
        task_id = os.path.basename(wav_path).split("_")[1].upper()

        row = metadata[metadata["RECODING ORIGINAL NAME"] == subject_id]
        if row.empty:
            continue

        sex = row["sex"].values[0]
        age = row["AGE"].values[0]
        label = row["label"].values[0]
        group_id = row["group_id"].values[0].upper()
        time_after_diagnosis = row["time after diagnosis"].values[0]
        updrs = row["UPDRS"].values[0]
        updrs_speech = row["UPDRS-speech"].values[0]
        hy_scale = row["H/Y"].values[0]

        dataset.append((subject_id, sample_id, task_id, label, group_id, sex, age,
                        updrs, updrs_speech, hy_scale, time_after_diagnosis))

    dataset_df = pd.DataFrame(dataset, columns=[
        "subject_id", "sample_id", "task_id", "label", "group_id",
        "sex", "age", "updrs_scale", "updrs_speech", "hy_scale", "time_after_diagnosis",
    ])

    # Paths to speech features (filled in after feature extraction)
    dataset_dir = os.path.sep.join(args.samples_dir.split(os.path.sep)[:-2])
    for feature_type in ["disvoice/articulation", "disvoice/glottal", "disvoice/phonation",
                         "disvoice/prosody", "wav2vec/layer07"]:
        feature_samples = []
        for _, sample in dataset_df.iterrows():
            sample_path = os.path.join(dataset_dir, "speech_features", feature_type,
                                       f'{sample["sample_id"]}.npz')
            feature_samples.append(sample_path)
        col = feature_type.replace("/", "").replace("disvoice", "").replace("layer07", "")
        dataset_df[col] = feature_samples

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "dataset.csv")
    dataset_df.to_csv(out_path)
    print(f"Dataset saved to {out_path} ({len(dataset_df)} samples)")
    print(dataset_df["task_id"].value_counts())
    print(dataset_df["group_id"].value_counts())
