# Whisper Transcription Service

This project provides a web-based transcription service using OpenAI's Whisper model. Users can upload audio files, and the service will transcribe the speech to text.

## Features

- Upload audio files via a web interface.
- Transcribe speech to text using Whisper.
- Supports various audio formats.
- Resamples audio to the required 16000 Hz for the Whisper model.

## Requirements

- Python 3.8 or higher
- `ffmpeg` installed and added to the system PATH

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/whisper-transcription-service.git
    cd whisper-transcription-service
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the Whisper model**:
    Ensure you have the Whisper model files in the `whisper/` directory. You can use the following script to download the model:

    ```python
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    processor.save_pretrained("./whisper")
    model.save_pretrained("./whisper")
    ```

5. **Ensure `ffmpeg` is installed**:
    ```bash
    ffmpeg -version
    ```

## Usage

1. **Run the Flask application**:
    ```bash
    python app.py
    ```

2. **Access the web application**:
    Open your web browser and go to `http://localhost:5000`.

3. **Upload an audio file**:
    - Select an audio file from your system.
    - Upload the file and view the transcription result.

## Project Structure

whisper-transcription-service/
├── app.py
├── transcribe.py
├── whisper/ # Contains the Whisper model files
├── uploads/ # Contains uploaded audio files
├── README.md
└── requirements.txt

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or bug reports.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
