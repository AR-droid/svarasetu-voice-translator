"""Build an evaluation manifest from FLEURS, whose sentences are parallel across
languages: the same sentence id read in the target language gives a reference
translation for free.

Usage: python -m voicenote.eval.make_fleurs_manifest --src ta --tgt hi --n 50 --out data/eval
"""

import argparse
import csv
import os

from voicenote.audio import write_wav
from voicenote.languages import get_language


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="data/eval")
    args = parser.parse_args()

    from datasets import load_dataset

    src, tgt = get_language(args.src), get_language(args.tgt)
    load = lambda code: load_dataset("google/fleurs", f"{code}_in", split=args.split, trust_remote_code=True)
    src_ds, tgt_ds = load(src.code), load(tgt.code)
    references = {row["id"]: row["raw_transcription"] for row in tgt_ds.remove_columns("audio")}

    audio_dir = os.path.join(args.out, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    manifest = os.path.join(args.out, f"fleurs_{src.code}-{tgt.code}.csv")
    seen, written = set(), 0
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "src_lang", "tgt_lang", "ref_translation"])
        for row in src_ds:
            # One recording per sentence keeps sentences (and mostly speakers) distinct.
            if row["id"] not in references or row["id"] in seen:
                continue
            seen.add(row["id"])
            name = f"{src.code}_{row['id']}.wav"
            write_wav(os.path.join(audio_dir, name), row["audio"]["array"], row["audio"]["sampling_rate"])
            writer.writerow([os.path.join("audio", name), src.code, tgt.code, references[row["id"]]])
            written += 1
            if written >= args.n:
                break
    print(f"Wrote {written} rows to {manifest}")


if __name__ == "__main__":
    main()
