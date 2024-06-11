import sqlite3
import torch

def add_speaker_embedding(name, embedding):
    conn = sqlite3.connect('speakers.db')
    c = conn.cursor()
    c.execute("INSERT INTO speakers (name, embedding) VALUES (?, ?)", (name, embedding))
    conn.commit()
    conn.close()

def get_all_speaker_embeddings():
    conn = sqlite3.connect('speakers.db')
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM speakers")
    results = c.fetchall()
    conn.close()
    
    speakers = []
    for name, embedding in results:
        tensor_embedding = torch.frombuffer(embedding, dtype=torch.float32)
        speakers.append((name, tensor_embedding))
    return speakers
