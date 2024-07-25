from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model and label encoder
with open('svr_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    species = request.form['species']
    length1 = float(request.form['length1'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])

    # Encode the species
    species_encoded = label_encoder.transform([species])[0]

    # Prepare the features
    features = np.array([[species_encoded, length1, length2, length3, height, width]])

    # Predict the weight
    prediction = model.predict(features)

    return render_template('index.html', prediction_text='Predicted Weight: {:.2f} grams'.format(prediction[0]))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
