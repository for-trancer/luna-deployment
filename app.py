from flask import Flask, request, jsonify, send_file
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          TextClassificationPipeline, AutoModelForTokenClassification, 
                          TokenClassificationPipeline)
from diffusers import LCMScheduler, AutoPipelineForText2Image
import torch
import base64
from io import BytesIO
from PIL import Image
import threading
import time

app = Flask(__name__)

# Model IDs from Hugging Face
model_intent_id = 'XLMRoberta-Alexa-Intents-Classification'
model_ner_id = 'XLMRoberta-Alexa-Intents-NER-NLU'
# For text-to-image generation:
base_model_id = "Lykon/dreamshaper-7"
lora_weights_id = "latent-consistency/lcm-lora-sdv1-5"  # Now loaded directly from HF

# --- Load the Intents Model ---
tokenizer_intent = AutoTokenizer.from_pretrained(model_intent_id)
model_intent = AutoModelForSequenceClassification.from_pretrained(model_intent_id)
classifier_intent = TextClassificationPipeline(model=model_intent, tokenizer=tokenizer_intent)

# --- Load the NER Model ---
tokenizer_ner = AutoTokenizer.from_pretrained(model_ner_id)
model_ner = AutoModelForTokenClassification.from_pretrained(model_ner_id)
classifier_ner = TokenClassificationPipeline(model=model_ner, tokenizer=tokenizer_ner)

# --- Load the Text-to-Image Generation Model ---
# Load the base pipeline from the Hugging Face hub
pipe = AutoPipelineForText2Image.from_pretrained(base_model_id, torch_dtype=torch.float32)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cpu")

# Load and fuse the online LoRA weights from Hugging Face
try:
    pipe.load_lora_weights(lora_weights_id)
    pipe.fuse_lora()
except ValueError as e:
    print("Error loading LoRA weights. Make sure PEFT is installed.")
    raise e

# --- Routes ---

# Intent Identification Endpoint
@app.route('/predict_intent', methods=['POST'])
def predict_intent():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    prediction = classifier_intent(text)
    print(prediction)
    return jsonify(prediction)

# Image Generation Endpoint
@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = data.get('text', '')
    with torch.no_grad():
        image = pipe(prompt=prompt, num_inference_steps=5, guidance_scale=0).images[0]
    # Convert the image to a base64 string (avoid image.show() in headless environments)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = jsonify({"image": f"data:image/png;base64,{img_str}"})
    response.headers['Connection'] = 'keep-alive'
    return response

# NER Endpoint
@app.route('/get_data', methods=['POST'])
def get_data():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    ner_results = classifier_ner(text)
    filtered_data = [
        {
            'entity': pred['entity'],
            'index': pred['index'],
            'word': pred['word'],
            'start': pred['start'],
            'end': pred['end']
        }
        for pred in ner_results
    ]
    print(filtered_data)
    return jsonify(filtered_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
