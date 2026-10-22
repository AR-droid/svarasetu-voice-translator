"""Run the pipeline over a manifest and report quality and speed.

Manifest: CSV with columns audio_path, src_lang, tgt_lang and optionally
ref_translation (a human translation, e.g. from FLEURS' parallel sentences).

Per note it measures
- round-trip CER: run ASR on the synthesised audio in the target language and
  compare with the text we asked TTS to say (intelligibility of the clone);
- speaker similarity: ECAPA cosine between the input note and the output, with
  a cross-speaker baseline (output vs other inputs) to show the cloning gain;
- chrF of the translation when a reference translation is given;
- per-stage latency and real-time factor.

Usage: python -m voicenote.eval.run_eval manifest.csv --out results/
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from voicenote.audio import ASR_SR, load_audio, resample, write_wav
from voicenote.eval.metrics import SpeakerEncoder, cer, chrf, cosine
from voicenote.pipeline import VoiceNoteTranslator, build_default_pipeline


def read_manifest(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    base = os.path.dirname(os.path.abspath(path))
    for row in rows:
        if not os.path.isabs(row["audio_path"]):
            row["audio_path"] = os.path.join(base, row["audio_path"])
    return rows


def evaluate(rows: list[dict], pipeline: VoiceNoteTranslator, encoder, out_dir: str) -> list[dict]:
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    records, in_embs, out_embs = [], [], []
    for i, row in enumerate(rows):
        wav = load_audio(row["audio_path"], ASR_SR)
        result = pipeline.translate(wav, row["src_lang"], row["tgt_lang"])
        out_16k = resample(result.audio, result.sample_rate, ASR_SR)
        name = f"{i:04d}_{result.src_lang}-{result.tgt_lang}.wav"
        write_wav(os.path.join(out_dir, "audio", name), result.audio, result.sample_rate)

        back = pipeline.asr.transcribe(out_16k, result.tgt_lang)
        in_embs.append(encoder.embed(wav))
        out_embs.append(encoder.embed(out_16k))
        record = {
            "audio_path": row["audio_path"],
            "output": name,
            "pair": f"{result.src_lang}-{result.tgt_lang}",
            "transcript": result.transcript,
            "translation": result.translation,
            "round_trip_asr": back,
            "round_trip_cer": round(cer(result.translation, back), 4),
            "speaker_sim": round(cosine(in_embs[-1], out_embs[-1]), 4),
            "input_s": round(result.input_seconds, 2),
            "output_s": round(result.output_seconds, 2),
            "rtf": round(result.real_time_factor, 3),
            **{f"t_{stage}": t for stage, t in result.timings.items()},
        }
        if row.get("ref_translation"):
            record["chrf"] = round(chrf(row["ref_translation"], result.translation), 2)
        records.append(record)
        print(f"[{i + 1}/{len(rows)}] {record['pair']} sim={record['speaker_sim']} cer={record['round_trip_cer']}")

    # Cross-speaker baseline: similarity of each output to every *other* input.
    if len(records) > 1:
        for i, record in enumerate(records):
            others = [cosine(out_embs[i], e) for j, e in enumerate(in_embs) if j != i]
            record["baseline_sim"] = round(float(np.mean(others)), 4)
    return records


def summarize(records: list[dict]) -> str:
    groups = defaultdict(list)
    for r in records:
        groups[r["pair"]].append(r)
    groups["all"] = records

    def mean(rows, key):
        values = [r[key] for r in rows if key in r]
        return f"{np.mean(values):.3f}" if values else "-"

    lines = [
        "| pair | n | speaker sim | cross-speaker baseline | round-trip CER | chrF | RTF | total latency (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pair, rows in groups.items():
        latency = f"{np.mean([sum(v for k, v in r.items() if k.startswith('t_')) for r in rows]):.2f}"
        lines.append(
            f"| {pair} | {len(rows)} | {mean(rows, 'speaker_sim')} | {mean(rows, 'baseline_sim')} | "
            f"{mean(rows, 'round_trip_cer')} | {mean(rows, 'chrf')} | {mean(rows, 'rtf')} | {latency} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest")
    parser.add_argument("--out", default="results")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)[: args.limit]
    records = evaluate(rows, build_default_pipeline(), SpeakerEncoder(), args.out)

    fields = list(dict.fromkeys(k for r in records for k in r))
    with open(os.path.join(args.out, "results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
