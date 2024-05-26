# Imports
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify

# Créer une application Flask
app = Flask(__name__)

# Importer modèle entrainé
with open('utils/best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

scaler = MinMaxScaler(feature_range = (0, 1))

@app.route('/')
def home_page():
    return 'Welcome to the credit scoring API'

# Dans onglet "Prediction" du site, méthode post pour envoyer une information à l'api (data) et récupérer prédiction
@app.route('/Prediction', methods = ['POST'])
def predict():
    try:
        # récupérer les données
        data = request.get_json(force = True)
        test_data = np.array(data['test_data'])

        # scaling des données
        scaled_data = scaler.transform(test_data)

        # predict
        prediction = model.predict(scaled_data)

        return jsonify({'prediction': prediction.tolist()}) # return le résultat dans un dictionnaire - tolist car ne prend pas les np arrays

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug = True)