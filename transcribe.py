import whisper
from pyannote.audio import Pipeline
import os
from transformers import pipeline
from flask import Flask, request, jsonify
import tempfile
import torchaudio

app = Flask(__name__)

# Paths to the models
whisper_model_path = "C:/Users/Eric/Projects/WhisperTranscription/models/large-v2"
sentiment_model_path = "C:/Users/Eric/Projects/WhisperTranscription/models/distilbert-base-uncased-finetuned-sst-2-english"
diarization_model_path = "C:/Users/Eric/Projects/WhisperTranscription/models/pyannote-segmentation"

# Check if models exist
print(f"Whisper model path {whisper_model_path} {'exists' if os.path.exists(whisper_model_path) else 'does not exist'}.")
print(f"Sentiment model path {sentiment_model_path} {'exists' if os.path.exists(sentiment_model_path) else 'does not exist'}.")
print(f"Diarization model path {diarization_model_path} {'exists' if os.path.exists(diarization_model_path) else 'does not exist'}.")

# Load the Whisper model
whisper_model = whisper.load_model("large-v2")

# Load the sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model_path)

# Load the diarization model
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")

@app.route('/', methods=['GET'])
def home():
    return """
    <!doctype html>
    <title>Transcription, Diarization, and Sentiment Analysis Service</title>
    <h1>Upload an audio file</h1>
    <form method="POST" action="/transcribe" enctype="multipart/form-data">
      <input type="file" name="audio" accept="audio/*">
      <input type="submit" value="Upload">
    </form>
    """

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Save the audio file to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        audio_path = os.path.join(tmpdirname, audio_file.filename)
        audio_file.save(audio_path)

        # Ensure the audio file is seekable by re-saving it with torchaudio
        seekable_audio_path = os.path.join(tmpdirname, "seekable_audio.wav")
        waveform, sample_rate = torchaudio.load(audio_path)
        torchaudio.save(seekable_audio_path, waveform, sample_rate)

        # Transcribe the audio
        result = whisper_model.transcribe(seekable_audio_path)
        transcription_text = result['text']

        # Perform speaker diarization
        diarization_result = diarization_pipeline({'uri': 'audio', 'audio': seekable_audio_path})

        # Perform sentiment analysis
        sentiments = sentiment_pipeline(transcription_text)

    # Convert diarization result to a more readable format
    speakers = []
    for segment, _, label in diarization_result.itertracks(yield_label=True):
        speakers.append({
            'start': segment.start,
            'end': segment.end,
            'speaker': label
        })

    return jsonify({
        'transcription': transcription_text,
        'diarization': speakers,
        'sentiments': sentiments
    })

if __name__ == '__main__':
    app.run(debug=True)
