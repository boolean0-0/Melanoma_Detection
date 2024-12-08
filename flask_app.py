from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA

from keys import keys

app = Flask(__name__)



# load models
models = {'CNN1': None, 'CNN2': None, 'SGD': None, 'PCA': None}
with open('models/CNN1.pkl', 'rb') as f:
    models['CNN1'] = pickle.load(file=f)
with open('models/CNN2.pkl', 'rb') as f:
    models['CNN2'] = pickle.load(file=f)
with open('models/SGD.pkl', 'rb') as f:
    models['SGD'] = pickle.load(file=f)
with open('models/pca_file.pkl', 'rb') as f:
    models['PCA'] = pickle.load(file=f)

@app.route('/')
def home():
    return 'API'
    
# Endpoint for making predictions
@app.route('/upload', methods=["POST", "GET"])
def predict():
    if request.method == 'POST':
        selected_model = request.args.get('model')
        uploaded_image = request.files.get('image')

        if uploaded_image != None:
            if '.jpg' in uploaded_image.filename:
                uploaded_image = Image.open(uploaded_image)
                uploaded_image = uploaded_image.resize((64, 64))
                uploaded_image = np.array(uploaded_image)
                uploaded_image_cnn = np.expand_dims(uploaded_image, axis = 0)
                uploaded_image_flattened = uploaded_image.reshape(-1)
                uploaded_image_flattened = uploaded_image_flattened.reshape(1, -1)
                uploaded_image_flattened_whitened = models['PCA'].transform(uploaded_image_flattened)
                n_components = models['PCA'].n_components_
                uploaded_image_whitened_reshaped = uploaded_image_flattened_whitened.reshape(1, n_components)
            else:
                return f"{uploaded_image.filename} is not a jpg image. Upload a file with the \'.jpg\' file extension."
        else:
            return "No image file uploaded", 400
        
        
        if str(selected_model) == "SGD":
            return jsonify({
                'prediction': str(models[str(selected_model)].predict(uploaded_image_whitened_reshaped))
            })
        elif str(selected_model) == "CNN1" or str(selected_model) == "CNN2":
            return jsonify({
                'prediction': str(models[str(selected_model)].predict(uploaded_image_cnn))
            })
    return 'error'


if __name__ == '__main__':
    app.run(debug=True)
