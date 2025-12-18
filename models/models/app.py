import tensorflow as tf
import pickle
import numpy as np
import re
import gradio as gr
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model("models/models/hindi_poem_model.keras")

with open("models/models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/models/max_len.txt", "r") as f:
    max_len = int(f.read())

def preprocess_poem(text):
    text = text.replace("\n", " <NL> ")
    text = text.replace("।", " <DANDA> ")
    text = re.sub(r"[^\u0900-\u097F <>A-Z_]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def decode_poem(text):
    return text.replace("<NL>", "\n").replace("<DANDA>", "।")

def sample_with_temperature(preds, temperature=0.8):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-9) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

def generate_poem(seed, temperature, length):
    seed = preprocess_poem(seed)

    seq = tokenizer.texts_to_sequences([seed])
    if len(seq[0]) == 0:
        return "Please enter a valid Hindi seed text."

    for _ in range(length):
        seq = tokenizer.texts_to_sequences([seed])[0]
        seq = pad_sequences([seq], maxlen=max_len, padding="pre")

        preds = model.predict(seq, verbose=0)[0]
        next_idx = sample_with_temperature(preds, temperature)
        next_word = tokenizer.index_word.get(next_idx, "")

        seed += " " + next_word

    return decode_poem(seed)

demo = gr.Interface(
    fn=generate_poem,
    inputs=[
        gr.Textbox(label="Seed line (Hindi)", placeholder="निर्माण फिरफिर नेह का"),
        gr.Slider(0.3, 1.2, value=0.8, step=0.1, label="Creativity (Temperature)"),
        gr.Slider(20, 150, value=70, step=10, label="Poem Length (words)")
    ],
    outputs=gr.Textbox(lines=12, label="Generated Poem"),
    title="Hindi Poem Generator",
)

demo.launch()
