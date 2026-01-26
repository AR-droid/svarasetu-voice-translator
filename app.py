import gradio as gr
from services.rag_pipeline import run_rag_pipeline
import os

def chatbot_response(audio, language_code):
    if audio is None:
        return "Please upload or record an audio file."
    
    # Gradio provides the path to the audio file
    result = run_rag_pipeline(audio, language_code)
    
    if isinstance(result, str):
        return result
    
    formatted_response = f"""
    **Transcription ({language_code}):** {result['transcript']}
    
    **Translated Query:** {result['english_query']}
    
    **Auto-Extracted Topic:** {result['topic']}
    
    **Answer:**
    {result['answer']}
    """
    return formatted_response

# Define the UI
with gr.Blocks(title="Indic Voice RAG Chatbot") as demo:
    gr.Markdown("# 🎙️ Indic Voice-Enabled RAG Chatbot")
    gr.Markdown("Speak your question in an Indian language. The system will automatically identify the topic, search Wikipedia, and answer you using RAG!")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record/Upload Audio")
            lang_input = gr.Dropdown(
                choices=["hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN", "bn-IN", "mr-IN", "gu-IN"],
                value="hi-IN",
                label="Select Language"
            )
            submit_btn = gr.Button("Ask Chatbot")
            
        with gr.Column():
            text_output = gr.Markdown(label="Response")

    submit_btn.click(
        fn=chatbot_response,
        inputs=[audio_input, lang_input],
        outputs=text_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
