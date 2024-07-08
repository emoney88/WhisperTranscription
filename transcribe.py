from flask import Flask, request, jsonify
from transformers import pipeline
from pyannote.audio import Pipeline
import os
import torch
import torchaudio
from io import BytesIO
import whisper
import html

app = Flask(__name__)

# Load pre-trained pipelines
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Load Whisper model
whisper_model = whisper.load_model("large-v2")  # Change to "large-v3" if necessary

# Function to perform diarization
def diarize_audio(file_path):
    # Load audio file ensuring it is seekable
    waveform, sample_rate = torchaudio.load(file_path)
    temp_buffer = BytesIO()
    torchaudio.save(temp_buffer, waveform, sample_rate, format="wav")
    temp_buffer.seek(0)
    
    diarization_result = diarization_pipeline(temp_buffer)
    segments = []
    for segment in diarization_result.itersegments():
        try:
            speaker_label = diarization_result[segment].label
        except KeyError:
            speaker_label = "unknown"
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker_label
        })
    return segments

# Function to identify speaker names in the transcription
def identify_speaker_names(transcription, diarization):
    speaker_names = {}
    words = transcription.split()
    for i, word in enumerate(words):
        if word.endswith(","):
            speaker_name = word.rstrip(",")
            next_word = words[i + 1] if i + 1 < len(words) else None
            if next_word and next_word.lower() in ["says", "said", "speaks"]:
                speaker_names[speaker_name] = "SPEAKER_{}".format(len(speaker_names) + 1)
    return speaker_names

# Function to update diarization segments with speaker names
def update_diarization_with_names(diarization, speaker_names, transcription_segments):
    updated_diarization = []
    for segment in diarization:
        speaker = segment['speaker']
        if speaker in speaker_names:
            segment['speaker'] = speaker_names[speaker]
        
        # Assign text to each segment
        segment_text = []
        for trans_segment in transcription_segments:
            if trans_segment['start'] >= segment['start'] and trans_segment['start'] < segment['end']:
                segment_text.append(trans_segment['text'])
        segment['text'] = ' '.join(segment_text).strip() or "[No text found]"
        updated_diarization.append(segment)
    return updated_diarization

# Function to generate HTML response
def generate_html(transcription, sentiment, diarization):
    html_content = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Transcription Results</title>
    </head>
    <body>
        <h1>Transcription Results</h1>
        <h2>Transcription</h2>
        <p>{html.escape(transcription)}</p>
        <h2>Sentiment</h2>
        <p>Label: {html.escape(sentiment['label'])}</p>
        <p>Score: {sentiment['score']:.2f}</p>
        <h2>Diarization</h2>
        <table border="1">
            <thead>
                <tr>
                    <th>Start</th>
                    <th>End</th>
                    <th>Speaker</th>
                    <th>Text</th>
                </tr>
            </thead>
            <tbody>
    """
    for entry in diarization:
        html_content += f"""
                <tr>
                    <td>{entry['start']:.2f}</td>
                    <td>{entry['end']:.2f}</td>
                    <td>{html.escape(entry['speaker'])}</td>
                    <td>{html.escape(entry['text'])}</td>
                </tr>
        """
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    return html_content

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Perform transcription using Whisper
    transcription_result = whisper_model.transcribe(file_path)
    transcription = transcription_result['text']

    # Extract segments from transcription
    transcription_segments = []
    for segment in transcription_result['segments']:
        transcription_segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text']
        })

    # Perform sentiment analysis
    sentiment_result = sentiment_pipeline(transcription)[0]

    # Perform diarization
    diarization_segments = diarize_audio(file_path)

    # Identify and update speaker names
    speaker_names = identify_speaker_names(transcription, diarization_segments)
    updated_diarization = update_diarization_with_names(diarization_segments, speaker_names, transcription_segments)

    # Generate HTML response
    html_response = generate_html(transcription, sentiment_result, updated_diarization)

    return html_response

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
