import requests

url = "http://0.0.0.0:8000/transcribe"
audio_path = "shubh_tts_audio.mp3"

files = {
    'audio': open(audio_path, 'rb')
}
data = {
    'language': 'hi' # Using 'hi' for Hindi transcription
}

print(f"Testing ASR API with {audio_path}...")
try:
    response = requests.post(url, files=files, data=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
