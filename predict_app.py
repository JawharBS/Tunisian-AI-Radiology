import base64
import numpy as np
import io
from PIL import Image
from flask import request
from flask import jsonify
from flask import Flask
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)

def get_model():
    global model
    model = load_model('vgg19.h5')
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    return image

print(" * Loading Keras model...")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    print('enter predict')
    data = []
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))        
    processed_image = preprocess_image(image, target_size=(224, 224))
    data.append(processed_image)
    data = np.array(data) / 255.0
    
    prediction = model.predict(data)
    print ('predict OK')
    print ('COVID-19 case = ' + str(prediction[0][0]*100) + '%')
    print ('Normal case = ' + str(prediction[0][1]*100) + '%')

    covid = (prediction[0][0].item())*100
    non_covid = (prediction[0][1].item())*100
    response = {
        'prediction': {
            'covid': covid,
            'non_covid': non_covid
        }
    }
    return jsonify(response)
