import os
import requests
from dotenv import load_dotenv
from services.translation import translate_to_english
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# Dynamic imports for Phase 1 & 2
import phase1_data.datacollection as datacollection
import phase2_vector.chunking as chunking
import phase2_vector.embeddings as embeddings_builder

load_dotenv()

# ---------------- CONFIG ---------------- #
ASR_API_URL = "http://0.0.0.0:8000/transcribe"
VECTOR_DB_DIR = "phase2_vector/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Config (Using Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile" 
LLM_BASE_URL = "https://api.groq.com/openai/v1"

# ---------------- UTILS ---------------- #

def transcribe_audio(audio_path: str, language: str = "hi") -> str:
    """Task 3: Call the deployed ASR endpoint"""
    try:
        with open(audio_path, 'rb') as f:
            files = {'audio': f}
            data = {'language': language}
            response = requests.post(ASR_API_URL, files=files, data=data)
            response.raise_for_status()
            return response.json().get("transcription", "")
    except Exception as e:
        print(f"ASR Error: {e}")
        return ""

def update_knowledge_base(topic: str):
    """Task 1 & 2: Dynamic Knowledge Collection and Indexing"""
    print(f"[0/4] Dynamically updating knowledge for topic: {topic}...")
    
    # Phase 1: Data Collection
    title = datacollection.search_wikipedia(topic)
    text = datacollection.fetch_full_article(title)
    cleaned = datacollection.clean_text(text)
    datacollection.save_text(cleaned, title)
    
    # Phase 2: Chunking
    filepath = os.path.join("data/raw", title.replace(" ", "_") + ".txt")
    chunking.run_chunking(filepath)
    
    # Phase 2: Embeddings
    chunks = embeddings_builder.load_chunks("phase2_vector/chunks.json")
    docs = embeddings_builder.build_documents(chunks)
    vectorstore = embeddings_builder.create_vector_db(docs)
    embeddings_builder.save_vector_db(vectorstore, docs)
    print(f"[0/4] Knowledge base updated with canonical article: {title}")

def retrieve_context(query: str, k: int = 2):
    """Task 2: Retrieve top-k chunks from Vector DB"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def generate_answer(question: str, context: str) -> str:
    """Task 5: Use LLM to get the final answer"""
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not found. Please provide it in .env"

    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=GROQ_API_KEY
    )

    prompt = f"""
    Answer the following question based ONLY on the provided context.
    If the context doesn't contain the answer, say you don't know.

    Context:
    {context}

    Question: {question}
    Answer:
    """

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"LLM Generation Error: {e}"

def extract_topic_keyword(query: str) -> str:
    """Extra Phase: Extract main subject keyword from query using LLM"""
    if not GROQ_API_KEY:
        return ""

    client = OpenAI(base_url=LLM_BASE_URL, api_key=GROQ_API_KEY)
    
    prompt = f"""
    Extract the main subject or entity from the following question to be used for a Wikipedia search.
    Output ONLY the keyword or a short phrase, nothing else.
    
    Question: {query}
    Topic Keyword:
    """
    
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20
        )
        keyword = completion.choices[0].message.content.strip()
        # Remove common fluff
        keyword = keyword.replace('"', '').replace("'", "")
        return keyword
    except Exception as e:
        print(f"Topic Extraction Error: {e}")
        return ""

# ---------------- PIPELINE ---------------- #

def run_rag_pipeline(audio_path: str, language: str = "hi"):
    print(f"--- Starting RAG Pipeline ---")
    
    # 1. Transcribe
    print(f"[1/5] Transcribing audio ({language})...")
    transcript = transcribe_audio(audio_path, language)
    if not transcript:
        return "Failed to transcribe audio."
    print(f"Transcript: {transcript}")

    # 2. Translate
    print(f"[2/5] Translating to English...")
    english_query = translate_to_english(transcript, f"{language}-IN" if '-' not in language else language)
    if english_query == transcript and language != "en":
        print("WARNING: Translation failed or returned original text.")
    print(f"English Query: {english_query}")

    # 3. Extra Phase: Extract Topic & Update Knowledge
    print(f"[3/5] Extracting topic and updating knowledge base...")
    topic = extract_topic_keyword(english_query)
    if topic:
        print(f"Extracted Topic: {topic}")
        update_knowledge_base(topic)
    else:
        print("Skipping knowledge update: No topic extracted.")

    # 4. Retrieve
    print(f"[4/5] Retrieving context from Vector DB...")
    context = retrieve_context(english_query)

    # 5. Generate
    print(f"[5/5] Generating answer using LLM...")
    answer = generate_answer(english_query, context)
    
    return {
        "transcript": transcript,
        "english_query": english_query,
        "topic": topic,
        "answer": answer
    }

if __name__ == "__main__":
    # Example usage
    # result = run_rag_pipeline("path/to/audio.wav", "hi")
    # print(f"\nFINAL ANSWER:\n{result['answer']}")
    pass
