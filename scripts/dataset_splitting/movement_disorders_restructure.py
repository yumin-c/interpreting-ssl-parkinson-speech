import os
import re
import glob
import shutil
import argparse
import pandas as pd
from tqdm import tqdm

# Maps Kaggle HC/PD subject IDs to AVPEPUDEA-format IDs expected by GITA pipeline
# HC{N} -> AVPEPUDEAC{N:04d}  (label=0)
# PD{N} -> AVPEPUDEA{N:04d}   (label=1)

VOWEL_TASK = "SUSTAINED-VOWELS"

def subject_to_gita_id(prefix, num):
    n = int(num)
    if prefix == "HC":
        return f"AVPEPUDEAC{n:04d}"
    else:
        return f"AVPEPUDEA{n:04d}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Restructure Movement Disorders (Kaggle) audio samples to GITA-compatible format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True, type=str,
                        help="Path to 'Archive Raw Audio' directory inside the dataset")
    parser.add_argument("--new-data-dir", default="./data/movement_disorders/audios/", type=str)
    parser.add_argument("--metadata-output", default="./data/movement_disorders/metadata.csv", type=str)
    parser.add_argument("--variant", default="clean", choices=["clean", "raw", "LF"],
                        help="Which file variant to use: 'clean', 'raw' (no suffix), or 'LF'")
    args = parser.parse_args()

    os.makedirs(args.new_data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metadata_output), exist_ok=True)

    hc_dir = os.path.join(args.data_dir, "Healthy a")
    pd_dir = os.path.join(args.data_dir, "Parkinson a")

    subjects = {}
    copied = 0
    skipped = 0

    for group_dir, group_prefix in [(hc_dir, "HC"), (pd_dir, "PD")]:
        wavs = glob.glob(os.path.join(group_dir, "*.wav"))
        for wav_path in tqdm(wavs, desc=f"Processing {group_prefix}"):
            fname = os.path.basename(wav_path).replace(".wav", "")

            # Parse: HC10a1, HC10a1_clean, HC10a1_LF
            m = re.match(r"(HC|PD)(\d+)([aeiou])(\d+)(?:_(.+))?$", fname, re.IGNORECASE)
            if not m:
                skipped += 1
                continue

            gp, subj_num, vowel, rep, suffix = m.groups()
            suffix = suffix or "raw"

            if args.variant == "clean" and suffix != "clean":
                continue
            if args.variant == "LF" and suffix != "LF":
                continue
            if args.variant == "raw" and suffix not in (None, "raw"):
                continue

            gita_id = subject_to_gita_id(gp, subj_num)
            group_id = gp.upper()
            vowel_upper = vowel.upper()
            sample_id = f"{gita_id}_{vowel_upper}{rep}"
            new_name = f"{group_id}_{VOWEL_TASK}_{sample_id}.wav"
            new_path = os.path.join(args.new_data_dir, new_name)
            shutil.copy(wav_path, new_path)
            copied += 1

            if gita_id not in subjects:
                subjects[gita_id] = {"group": group_id, "label": 0 if gp == "HC" else 1}

    print(f"Copied {copied} files, skipped {skipped} unrecognized files")

    # Build metadata CSV matching GITA format
    rows = []
    for gita_id, info in sorted(subjects.items()):
        rows.append({
            "RECODING ORIGINAL NAME": gita_id,
            "UPDRS": 0 if info["label"] == 0 else float("nan"),
            "UPDRS-speech": float("nan"),
            "H/Y": float("nan"),
            "SEX": "U",
            "AGE": 0,
            "time after diagnosis": 0,
        })

    metadata_df = pd.DataFrame(rows)
    metadata_df.to_csv(args.metadata_output, index=False)
    print(f"Metadata saved to {args.metadata_output} ({len(rows)} subjects)")
    print(f"  HC: {sum(1 for r in rows if r['RECODING ORIGINAL NAME'].startswith('AVPEPUDEAC'))}")
    print(f"  PD: {sum(1 for r in rows if not r['RECODING ORIGINAL NAME'].startswith('AVPEPUDEAC'))}")
