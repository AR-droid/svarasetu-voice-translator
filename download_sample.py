import requests
url = "https://www.kozco.com/tech/lrmono.wav"
r = requests.get(url)
with open("sample.wav", "wb") as f:
    f.write(r.content)
print("Downloaded sample.wav")
