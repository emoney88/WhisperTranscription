from flask import Flask, request, render_template_string
from transcribe import transcribe_audio
import os
import sqlite3

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Create a database to store speaker embeddings if it doesn't exist
def init_db():
    conn = sqlite3.connect('speakers.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS speakers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, embedding BLOB)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            transcription = transcribe_audio(file_path)
            return render_template_string(HTML_TEMPLATE, transcription=transcription)

    return render_template_string(HTML_TEMPLATE)

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcription Service</title>
</head>
<body>
    <h1>Upload an audio file for transcription</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="audio/*" required>
      <input type="submit" value="Upload">
    </form>
    {% if transcription %}
    <h2>Transcription:</h2>
    <pre>{{ transcription }}</pre>
    {% endif %}
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, port=5000)
