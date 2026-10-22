"""Gradio web demo: record a note, pick languages, hear it back in your own voice."""

import os

import gradio as gr

from voicenote.languages import LANGUAGES
from voicenote.pipeline import build_default_pipeline

CHOICES = [(f"{lang.name} ({lang.code})", lang.code) for lang in LANGUAGES.values()]
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_default_pipeline()
    return _pipeline


def run(audio_path, src_lang, tgt_lang, consent):
    if not consent:
        raise gr.Error("Please confirm the voice note is your own voice, or you have permission to clone it.")
    if not audio_path:
        raise gr.Error("Record or upload a voice note first.")
    try:
        result = get_pipeline().translate_file(audio_path, src_lang, tgt_lang)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    timings = "\n".join(
        [f"| {stage} | {seconds:.2f} s |" for stage, seconds in result.timings.items()]
        + [
            f"| **total** | **{result.total_seconds:.2f} s** |",
            f"| input length | {result.input_seconds:.1f} s |",
            f"| real-time factor | {result.real_time_factor:.2f} |",
        ]
    )
    return (
        (result.sample_rate, result.audio),
        result.transcript,
        result.translation,
        "| stage | time |\n| --- | --- |\n" + timings,
    )


with gr.Blocks(title="Voice Note Translator") as demo:
    gr.Markdown(
        "# Voice Note Translator\n"
        "Speak in one Indian language. Your friend hears it in theirs, **in your voice**.\n\n"
        "IndicConformer ASR → Indic-to-Indic translation → IndicF5 zero-shot voice cloning."
    )
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Your voice note")
            with gr.Row():
                src = gr.Dropdown(CHOICES, value="ta", label="You speak")
                tgt = gr.Dropdown(CHOICES, value="hi", label="They hear")
            consent = gr.Checkbox(label="This is my own voice, or I have permission to clone it")
            button = gr.Button("Translate", variant="primary")
        with gr.Column():
            audio_out = gr.Audio(label="Translated note (your voice)", type="numpy")
            transcript = gr.Textbox(label="What you said")
            translation = gr.Textbox(label="Translation")
            stats = gr.Markdown()
    button.click(run, [audio_in, src, tgt, consent], [audio_out, transcript, translation, stats])


if __name__ == "__main__":
    # GRADIO_SHARE=1 gives a temporary public *.gradio.live link (e.g. from Kaggle or Colab).
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=os.getenv("GRADIO_SHARE") == "1")
