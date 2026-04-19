from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from googletrans import Translator
import json
import time
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("saved/final_model.keras")

# Translator
translator = Translator()

# Load labels
with open("label_map.json") as f:
    label_map = json.load(f)

last_time = 0
COOLDOWN = 2


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    global last_time

    data = request.json

    if not data or "keypoints" not in data:
        return jsonify({"result": None, "english": None})

    keypoints = np.array(data["keypoints"], dtype=np.float32)

    if keypoints.shape != (20, 225):
        return jsonify({"result": None, "english": None})

    keypoints = np.expand_dims(keypoints, axis=0)

    preds = model.predict(keypoints, verbose=0)[0]

    top3 = np.argsort(preds)[-3:]
    class_id = int(np.random.choice(top3))

    sign_text = label_map[str(class_id)]

    now = time.time()
    if now - last_time < COOLDOWN:
        return jsonify({"result": None, "english": None})

    last_time = now

    target_lang = data.get("language", "en")

    try:
        translated = translator.translate(sign_text, dest=target_lang).text
    except:
        translated = sign_text

    return jsonify({
        "english": sign_text,
        "result": translated
    })


# 🔥 IMPORTANT PART (PORT FIX FOR RENDER)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
