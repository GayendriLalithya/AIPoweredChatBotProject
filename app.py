import os
import uuid
import datetime
from flask import Flask, render_template, request, jsonify
import pyttsx3
import speech_recognition as sr
import torch
import random
import json
from google.cloud import vision
import google.generativeai as genai
from utils.model_utils import load_model, load_label_encoder, load_vectorizer
from utils.data_utils import load_knowledge_base, find_product_details
from threading import Thread
import subprocess
import time
from PIL import Image
import io
from celery import Celery
import redis

app = Flask(__name__)

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Tokenization function
def tokenize(text):
    return text.split()

# Load model and preprocessors
model_path = 'models/intent_classifier.pth'
label_encoder_path = 'models/label_encoder.pkl'
vectorizer_path = 'models/vectorizer.pkl'
intents_path = 'data/intents.json'
knowledge_base_path = 'data/knowledge_base.json'
user_generated_intents_path = 'data/user_generated_intents.json'

label_encoder = load_label_encoder(label_encoder_path)
vectorizer = load_vectorizer(vectorizer_path, tokenize)
with open(intents_path, 'r') as f:
    intents = json.load(f)['intents']

try:
    knowledge_base = load_knowledge_base(knowledge_base_path)
except json.JSONDecodeError as e:
    print(f"Error loading knowledge base: {e}")
    knowledge_base = []

input_dim = len(vectorizer.get_feature_names_out())
output_dim = len(label_encoder.classes_)

model = load_model(model_path, input_dim, output_dim)
model.eval()

# Google Cloud Vision Configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ADD YOUR OWN APIKEYS MAN"
client = vision.ImageAnnotatorClient()

# Gemini API Keys Configuration
GENAI_API_KEY_QUERY = "ADD YOUR OWN APIKEYS MAN"
GENAI_API_KEY_IMAGE_VALIDATION = "ADD YOUR OWN APIKEYS MAN"

genai_query = genai.GenerativeModel('gemini-pro')
genai_validation = genai.GenerativeModel('gemini-pro')

# Celery Configuration
def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    return celery

app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)

celery = make_celery(app)

@celery.task
def search_product_in_inventory(web_entities, knowledge_base):
    for entity in web_entities:
        product_name = entity.description.strip()
        for item in knowledge_base:
            if item['name'].lower() == product_name.lower():
                return True, product_name
    return False, None

@celery.task
def get_gemini_product_details(product_details):
    genai.configure(api_key=GENAI_API_KEY_IMAGE_VALIDATION)
    return get_gemini_response(f"Is the following product tech-related? \n{product_details}", genai_validation)

def detect_web_entities(content):
    image = vision.Image(content=content)
    response = client.web_detection(image=image)
    web_entities = response.web_detection.web_entities
    return web_entities

def get_gemini_response(question, model):
    chat = model.start_chat(history=[])
    response = chat.send_message(question)
    return response.text

def get_response(user_input):
    genai.configure(api_key=GENAI_API_KEY_QUERY)
    X = vectorizer.transform([user_input]).toarray()
    inputs = torch.tensor(X, dtype=torch.float32)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    tag = label_encoder.inverse_transform(predicted.numpy())[0]
    return tag

def log_interaction(user_input, response, tag="user_generated"):
    new_data = {
        "patterns": [user_input],
        "responses": [response],
        "tag": tag
    }
    try:
        with open(user_generated_intents_path, 'r') as file:
            user_generated_intents = json.load(file)
    except FileNotFoundError:
        user_generated_intents = {"intents": []}

    # Check if the tag already exists
    for intent in user_generated_intents['intents']:
        if intent['tag'] == tag:
            intent['patterns'].append(user_input)
            intent['responses'].append(response)
            break
    else:
        user_generated_intents['intents'].append(new_data)

    with open(user_generated_intents_path, 'w') as file:
        json.dump(user_generated_intents, file, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/speak', methods=['POST'])
def speak():
    text = request.json.get('text')
    engine.say(text)
    engine.runAndWait()
    return jsonify({'status': 'success'})

@app.route('/listen', methods=['POST'])
def listen():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            return jsonify({'text': text})
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand the audio'})
    except sr.RequestError as e:
        return jsonify({'error': f'Could not request results; {e}'})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    tag = get_response(user_input)

    print(f"User input: {user_input}")
    print(f"Predicted tag: {tag}")

    response = "Sorry, I couldn't understand your query."
    if tag == 'product_query':
        product_name = user_input.split("about")[-1].strip()  # Extract product name from user input
        print(f"Product name extracted: {product_name}")
        response = find_product_details(product_name, knowledge_base)
        if response == "Sorry, I couldn't find any details for that product.":
            genai.configure(api_key=GENAI_API_KEY_QUERY)
            gemini_response = get_gemini_response(user_input, genai_query)
            response += f"\n\n(This information is gathered from Gemini API: {gemini_response})"
            log_interaction(user_input, response, tag="product_query")
    elif tag == 'noans':
        genai.configure(api_key=GENAI_API_KEY_QUERY)
        gemini_response = get_gemini_response(user_input, genai_query)
        if "tech" in gemini_response.lower() or "pc parts" in gemini_response.lower():
            response += f"\n\n(This information is gathered from Gemini API: {gemini_response})"
            gemini_tag = gemini_response.split("\n\n")[0].replace("**Tag:** ", "")
            log_interaction(user_input, response, tag=gemini_tag)
        else:
            response = "I'm sorry, I'm only allowed to answer tech-related and PC parts questions. :)"
            log_interaction(user_input, response, tag="non_tech_questions")
    else:
        for intent in intents:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                break
        log_interaction(user_input, response, tag=tag)

    print(f"Response: {response}")
    return jsonify({'response': response})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'response': "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'response': "No selected file"}), 400

    if file:
        img = Image.open(file.stream)
        img = img.convert('RGB')  # Ensure image is in RGB format

        # Generate unique filename using UUID
        unique_filename = str(uuid.uuid4())
        img_format = 'JPEG' if file.mimetype == 'image/jpeg' else 'PNG'
        img_filename = f"{unique_filename}.{img_format.lower()}"

        # Convert image to byte array for Google Cloud Vision
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img_format)
        img_byte_arr.seek(0)
        content = img_byte_arr.getvalue()

        # Detect web entities
        web_entities = detect_web_entities(content)
        product_details = ""
        if web_entities:
            for entity in web_entities:
                product_details += f"{entity.description}\n"

        # Start asynchronous tasks
        search_task = search_product_in_inventory.apply_async(args=[web_entities, knowledge_base])
        gemini_task = get_gemini_product_details.apply_async(args=[product_details])

        # Wait for task results
        product_available, product_name = search_task.get(timeout=10)
        gemini_response = gemini_task.get(timeout=10)
        tech_related = "yes" in gemini_response.lower()

        # Set folder and JSON file based on tech-related status
        if tech_related:
            folder = 'pc_related'
            json_file = 'uploads/responses/pc_related_intents.json'
            availability_message = "Available in inventory." if product_available else "Not available in inventory."
            response_message = f"The image is of a tech-related product. Details:\n{product_details}\nAvailability: {availability_message}"
        else:
            folder = 'non_pc_related'
            json_file = 'uploads/responses/non_pc_related_intents.json'
            response_message = "I'm sorry, the provided image is not tech-related. Please try uploading another one :)"

        # Save the image to the appropriate folder
        img_path = os.path.join('uploads', 'images', folder, img_filename)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img.save(img_path, format=img_format)

        # Save the response to the appropriate JSON file
        response_data = {
            'image_filename': img_filename,
            'response_message': response_message,
            'timestamp': datetime.datetime.now().isoformat()
        }
        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        # Load existing data if the file exists and is not empty
        if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
            with open(json_file, 'r') as response_file:
                image_intents = json.load(response_file)
        else:
            image_intents = {'intents': []}

        # Append the new response data
        image_intents['intents'].append(response_data)

        # Save the updated data back to the JSON file
        with open(json_file, 'w') as response_file:
            json.dump(image_intents, response_file, indent=4)

        return jsonify({'response': response_message})

def classify_image(image):
    # This function is now integrated with Google Cloud Vision, no need for the random choice
    pass

# Function to run the Jupyter notebook
def run_notebook():
    subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 'notebooks/train_model.ipynb'])

# Function to run the notebook periodically
def run_notebook_periodically(interval=600):
    while True:
        run_notebook()
        time.sleep(interval)

# Start the notebook execution thread
notebook_thread = Thread(target=run_notebook_periodically, args=(600,))
notebook_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
