import tempfile
import shutil

def save_uploaded_audio(upload_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with open(temp_file.name, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return temp_file.name
