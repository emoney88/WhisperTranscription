from flask import Flask, request, jsonify, render_template
import re
import whisper
import os
from pyannote.audio import Pipeline
from transformers import pipeline as transformers_pipeline
from pydub import AudioSegment

app = Flask(__name__)

# Define your speaker mapping
speaker_mapping = {
    "SPEAKER_00": "Unknown",
    "SPEAKER_01": "Unknown"
}

# Sample list of common names for simplicity. In a real application, you might use a more sophisticated method.
common_names = ["Mike", "Joe", "Eric"]

def detect_name(text):
    # Simple name detection using regex and common names list
    for name in common_names:
        if re.search(rf'\b{name}\b', text, re.IGNORECASE):
            return name
    return None

def replace_speaker_labels(transcription, mapping):
    for segment in transcription:
        speaker = segment["speaker"]
        detected_name = detect_name(segment["text"])
        if detected_name:
            mapping[speaker] = detected_name
        segment["speaker"] = mapping.get(speaker, speaker)
    return transcription

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Handle file upload and save it
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filepath = "./uploads/" + file.filename
    file.save(filepath)

    # Convert file to a seekable format using pydub
    audio = AudioSegment.from_file(filepath)
    seekable_filepath = filepath.replace("./uploads/", "./uploads/seekable_")
    audio.export(seekable_filepath, format="wav")

    # Load Whisper model and perform transcription
    model = whisper.load_model("large")
    result = model.transcribe(seekable_filepath)

    # Authenticate with Hugging Face
    hf_token = "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"
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=hf_token)

    # Diarization
    diarization = diarization_pipeline(seekable_filepath)

    # Prepare segments with diarization
    diarized_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarized_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    # Align transcription with diarization
    aligned_transcription = []
    for segment in result['segments']:
        text_segment = {
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text'],
            "speaker": "Unknown"
        }
        for diarized_segment in diarized_segments:
            if text_segment['start'] >= diarized_segment['start'] and text_segment['end'] <= diarized_segment['end']:
                text_segment['speaker'] = diarized_segment['speaker']
                break
        aligned_transcription.append(text_segment)

    # Replace generic speaker labels with specific names
    updated_transcription = replace_speaker_labels(aligned_transcription, speaker_mapping)

    # Sentiment Analysis
    sentiment_analysis_pipeline = transformers_pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    for segment in updated_transcription:
        sentiment = sentiment_analysis_pipeline(segment["text"])[0]
        segment["sentiment"] = {
            "label": sentiment["label"],
            "score": sentiment["score"]
        }

    # Return the updated transcription as JSON response
    return render_template('result.html', transcription=updated_transcription)

if __name__ == '__main__':
    if not os.path.exists('./uploads/'):
        os.makedirs('./uploads/')
    app.run(debug=True)
