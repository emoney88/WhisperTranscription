import os
import flask
import torchaudio
from transformers import pipeline
from pyannote.audio import Pipeline
import whisper

app = flask.Flask(__name__)

whisper_model_path = "large-v2"
diarization_model_path = "pyannote/speaker-diarization"
auth_token = "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"

# Load Whisper model
try:
    whisper_model = whisper.load_model(whisper_model_path)
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")

# Load Pyannote diarization pipeline
try:
    diarization_pipeline = Pipeline.from_pretrained(diarization_model_path, use_auth_token=auth_token)
    print("Diarization pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading diarization model: {e}")
    diarization_pipeline = None

# Load Sentiment Analysis pipeline
try:
    sentiment_pipeline = pipeline("sentiment-analysis")
    print("Sentiment analysis pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading sentiment analysis pipeline: {e}")
    sentiment_pipeline = None

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in flask.request.files:
        return "No file provided", 400
    
    audio_file = flask.request.files["file"]
    audio_path = os.path.join("uploads", audio_file.filename)
    audio_file.save(audio_path)

    # Perform transcription
    try:
        result = whisper_model.transcribe(audio_path)
        transcription = result["text"]
        segments = result["segments"]
        print(f"Transcription: {transcription}")
        print(f"Segments: {segments}")
    except Exception as e:
        return f"Error during transcription: {e}", 500

    # Perform sentiment analysis
    try:
        sentiment_result = sentiment_pipeline(transcription)
        sentiment_label = sentiment_result[0]['label']
        sentiment_score = sentiment_result[0]['score']
        print(f"Sentiment: {sentiment_label}, Score: {sentiment_score}")
    except Exception as e:
        return f"Error during sentiment analysis: {e}", 500

    # Perform diarization
    diarization_segments = []
    if diarization_pipeline:
        try:
            audio, sample_rate = torchaudio.load(audio_path)
            diarization_result = diarization_pipeline({"waveform": audio, "sample_rate": sample_rate})
            print("Diarization result:")
            print(diarization_result)

            # Create segments with corresponding text
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                start = turn.start
                end = turn.end
                text_segment = get_text_segment(segments, start, end)
                print(f"Start: {start}, End: {end}, Speaker: {speaker}, Text: {text_segment}")
                if text_segment:
                    diarization_segments.append({
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                        "text": text_segment
                    })
        except Exception as e:
            return f"Error during diarization: {e}", 500

    return flask.render_template("results.html", 
                                 transcription=transcription, 
                                 sentiment={"label": sentiment_label, "score": sentiment_score}, 
                                 diarization=diarization_segments)

def get_text_segment(segments, start, end):
    text = []
    for segment in segments:
        segment_start = segment["start"]
        segment_end = segment["end"]
        segment_text = segment["text"]
        
        # Check if the segment overlaps with the diarization interval
        if segment_end > start and segment_start < end:
            # Clip the segment text to the diarization interval
            if segment_start < start:
                clip_start = int((start - segment_start) * 1000)
            else:
                clip_start = 0

            if segment_end > end:
                clip_end = int((end - segment_start) * 1000)
            else:
                clip_end = len(segment_text)

            clipped_text = segment_text[clip_start:clip_end].strip()
            text.append(clipped_text)

    return " ".join(text).strip()

if __name__ == "__main__":
    app.run(debug=True)
