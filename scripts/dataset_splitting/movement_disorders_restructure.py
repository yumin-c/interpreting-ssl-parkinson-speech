import os
import re
import glob
import shutil
import argparse
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm

# Maps Kaggle HC/PD/AD subject IDs to internal IDs used by the pipeline
# HC{N}    -> AVPEPUDEAC{N:04d}   (label=0)
# PD{N}    -> AVPEPUDEA{N:04d}    (label=1)
# adrsoN   -> AVPEPUDEAAD{N:04d}  (label=2)

VOWEL_TASK = "SUSTAINED-VOWELS"
LABEL_MAP = {"HC": 0, "PD": 1, "AD": 2}

def subject_to_gita_id(prefix, num):
    n = int(num)
    if prefix == "HC":
        return f"AVPEPUDEAC{n:04d}"
    elif prefix == "PD":
        return f"AVPEPUDEA{n:04d}"
    else:  # AD
        return f"AVPEPUDEAAD{n:04d}"

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
    parser.add_argument("--onset-trim-sec", default=0.0, type=float,
                        help="Seconds to trim from the start of each file (removes evaluator instruction, default: 0)")
    args = parser.parse_args()

    os.makedirs(args.new_data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metadata_output), exist_ok=True)

    hc_dir = os.path.join(args.data_dir, "Healthy a")
    pd_dir = os.path.join(args.data_dir, "Parkinson a")
    ad_dir = os.path.join(args.data_dir, "Alzheimer")

    subjects = {}
    copied = 0
    skipped = 0

    # --- HC and PD (vowel recordings with repetitions) ---
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

            if args.onset_trim_sec > 0:
                audio, sr = sf.read(wav_path)
                trim_samples = int(args.onset_trim_sec * sr)
                sf.write(new_path, audio[trim_samples:], sr)
            else:
                shutil.copy(wav_path, new_path)
            copied += 1

            if gita_id not in subjects:
                subjects[gita_id] = {"group": group_id, "label": LABEL_MAP[group_id]}

    # --- Alzheimer (single recording per subject, adrsoXXX.wav) ---
    ad_wavs = sorted(glob.glob(os.path.join(ad_dir, "*.wav")))
    for wav_path in tqdm(ad_wavs, desc="Processing AD"):
        fname = os.path.basename(wav_path).replace(".wav", "")
        m = re.match(r"adrso(\d+)$", fname)
        if not m:
            skipped += 1
            continue

        subj_num = m.group(1)
        gita_id = subject_to_gita_id("AD", subj_num)
        sample_id = f"{gita_id}_R1"
        new_name = f"AD_{VOWEL_TASK}_{sample_id}.wav"
        new_path = os.path.join(args.new_data_dir, new_name)

        if args.onset_trim_sec > 0:
            audio, sr = sf.read(wav_path)
            trim_samples = int(args.onset_trim_sec * sr)
            sf.write(new_path, audio[trim_samples:], sr)
        else:
            shutil.copy(wav_path, new_path)
        copied += 1

        if gita_id not in subjects:
            subjects[gita_id] = {"group": "AD", "label": LABEL_MAP["AD"]}

    print(f"Copied {copied} files, skipped {skipped} unrecognized files")

    # Build metadata CSV — store label explicitly for 3-class support
    rows = []
    for gita_id, info in sorted(subjects.items()):
        rows.append({
            "RECODING ORIGINAL NAME": gita_id,
            "label": info["label"],
            "group_id": info["group"],
            "UPDRS": float("nan"),
            "UPDRS-speech": float("nan"),
            "H/Y": float("nan"),
            "SEX": "U",
            "AGE": 0,
            "time after diagnosis": 0,
        })

    metadata_df = pd.DataFrame(rows)
    metadata_df.to_csv(args.metadata_output, index=False)
    print(f"Metadata saved to {args.metadata_output} ({len(rows)} subjects)")
    for grp in ["HC", "PD", "AD"]:
        n = sum(1 for r in rows if r["group_id"] == grp)
        print(f"  {grp}: {n}")
