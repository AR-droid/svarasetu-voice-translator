import requests

url = "http://0.0.0.0:8000/transcribe"
audio_path = "shubh_tts_audio.mp3"

files = {
    'audio': open(audio_path, 'rb')
}
data = {
    'language': 'hi' 
}

print(f"Testing ASR API with {audio_path} (Language: hi)...")
try:
    response = requests.post(url, files=files, data=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Transcription: {response.json().get('transcription')}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {e}")
