import time
import whisper
from flask import Flask, render_template

app = Flask(__name__)

model = whisper.load_model('tiny.en')
file = "5.m4a"

@app.route("/")
def render():
    while True:
        text_return = model.transcribe(file, fp16=False)
        print(text_return['text'])
        time.sleep(5)

if __name__ == "__main__":
    app.run()
