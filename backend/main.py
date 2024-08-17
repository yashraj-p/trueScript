from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np
import zipfile
import os
import re
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import string
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from functools import wraps
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Initialize Flask app
app = Flask(__name__)

# Setup CORS
CORS(app,
     resources={r"/*": {"origins": "http://localhost:5000"}})  # This allows all origins. Adjust as needed for security.

# Global variables
stop_words = set(stopwords.words('english'))
scaler = StandardScaler()
tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_model'])
MAX_SEQ_LENGTH = config['max_seq_length']
loaded_model = None
inference_func = None

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


# Utility functions
def error_response(message, status_code):
    return jsonify({"error": message}), status_code


def success_response(data, status_code=200):
    return jsonify(data), status_code


def require_model(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if loaded_model is None or inference_func is None:
            return error_response("Model not initialized", 400)
        return f(*args, **kwargs)

    return decorated_function

# Model initialization
@app.route('/initialize_model', methods=['POST'])
def initialize_model():
    global loaded_model, inference_func
    model_zip_path = request.json.get('model_zip_path')
    model_extract_path = request.json.get('model_extract_path')

    if not model_zip_path or not model_extract_path:
        return error_response("Model zip path and extract path must be provided", 400)

    try:
        # Adjust the zip file path relative to the current directory
        zip_file_path = os.path.join(os.getcwd(), model_zip_path)

        if not os.path.exists(model_extract_path):
            # Extract the zip file into the extract path in the current directory
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(model_extract_path)

        loaded_model = tf.saved_model.load(model_extract_path)
        inference_func = loaded_model.signatures['serving_default']
        logger.info("Model loaded successfully with serving function")
        return success_response({"message": "Model initialized successfully"})
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return error_response(str(e), 500)


# Text classification
@app.route('/classify_text', methods=['POST'])
@require_model
def classify_text_endpoint():
    text = request.json.get('text')
    if not text:
        return error_response("Text must be provided", 400)

    try:
        classification_result = classify_text(text)
        return success_response({"classification": classification_result})
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        return error_response(str(e), 500)


# Text processing functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[<>\\/,\'"]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def count_syllables(word):
    return max(1, len(re.findall(r'[aeiou]', word, re.I)))


def calculate_perplexity(text):
    sentences = sent_tokenize(text)
    all_sentences = [word_tokenize(sentence) for sentence in sentences]

    bigrams = [(w1, w2) for sentence in all_sentences for w1, w2 in zip(sentence[:-1], sentence[1:])]
    bigram_freq = Counter(bigrams)
    unigram_freq = Counter([word for sentence in all_sentences for word in sentence])

    def perplexity_for_sentence(sentence):
        tokens = word_tokenize(sentence)
        bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        vocab_size = len(unigram_freq)
        probs = []
        for w1, w2 in bigrams:
            prob = (bigram_freq[(w1, w2)] + 1) / (unigram_freq[w1] + vocab_size)
            probs.append(prob)
        return np.exp(-np.mean(np.log(probs))) if probs else np.nan

    perplexities = [perplexity_for_sentence(sentence) for sentence in sentences]
    return np.nanmean(perplexities)


def calculate_additional_features(text):
    preprocessed_text = preprocess_text(text)
    words = word_tokenize(preprocessed_text)
    sentences = sent_tokenize(text)
    word_count = len(words)
    char_count = len(preprocessed_text)
    sentence_count = len(sentences)

    word_density = word_count / char_count if char_count > 0 else 0
    avg_line_length = np.mean([len(line) for line in text.split('\n')])

    word_freq = Counter(words)
    frequencies = np.array(list(word_freq.values()))
    mean_freq = np.mean(frequencies)
    variance = np.var(frequencies)
    burstiness_score = (variance - mean_freq) / (variance + mean_freq) if (variance + mean_freq) != 0 else 0

    syllables = [count_syllables(word) for word in words]
    num_syllables = sum(syllables)
    complex_words = sum(1 for s in syllables if s > 2)

    if sentence_count > 0 and word_count > 0:
        fk_grade = 0.39 * (word_count / sentence_count) + 11.8 * (num_syllables / word_count) - 15.59
        gunning_fog = 0.4 * ((word_count / sentence_count) + 100 * (complex_words / word_count))
    else:
        fk_grade = gunning_fog = 0

    mean_perplexity = calculate_perplexity(text)

    return [
        avg_line_length, word_density, mean_perplexity,
        burstiness_score, fk_grade, gunning_fog
    ]


def batch_tokenize(texts, tokenizer, batch_size=32, max_length=MAX_SEQ_LENGTH):
    input_ids_list = []
    attention_mask_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding='max_length', truncation=True, max_length=max_length,
                           return_tensors='tf')
        input_ids_list.append(inputs['input_ids'])
        attention_mask_list.append(inputs['attention_mask'])
    input_ids = tf.concat(input_ids_list, axis=0)
    attention_masks = tf.concat(attention_mask_list, axis=0)
    return input_ids, attention_masks


def predict(text, additional_features):
    input_ids, attention_masks = batch_tokenize([text], tokenizer)

    input_ids = tf.cast(input_ids, tf.int32)
    attention_masks = tf.cast(attention_masks, tf.int32)

    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'additional_features': tf.constant(additional_features.reshape(1, -1), dtype=tf.float32)
    }

    prediction = inference_func(**inputs)
    output_key = list(prediction.keys())[0]
    predicted_label = int((prediction[output_key] > 0.5).numpy().astype(int))

    return "AI Generated" if predicted_label == 1 else "Human Written"


def classify_text(text):
    features = calculate_additional_features(text)
    scaled_features = scaler.fit_transform(np.array(features).reshape(1, -1))
    result = predict(text, scaled_features)
    return result


if __name__ == '__main__':
    app.run(debug=config['debug'], host=config['host'], port=config['port'])