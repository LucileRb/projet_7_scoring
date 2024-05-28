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

# Importer scaler entrainé
with open('utils/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home_page():
    return 'Welcome to the credit scoring API'

# Dans onglet "Prediction" du site, méthode post pour envoyer une information à l'api (data) et récupérer prédiction
@app.route('/Prediction', methods = ['POST'])
def predict():
    try:
        # récupérer les données
        print('predict api')
        data = request.get_json(force = True)
        print(data)
        test_data = np.array(data['test_data'])
        print(test_data)

        # Vérifier les dimensions des données
        if test_data.ndim != 2:
            return jsonify({'error': 'Les données doivent être un tableau 2D'})

        # scaling des données
        scaled_data = scaler.transform(test_data)
        print(scaled_data)

        # predict
        print('prediction')
        prediction = model.predict(scaled_data)
        print(prediction)

        return jsonify({'prediction': prediction.tolist()}) # return le résultat dans un dictionnaire - tolist car ne prend pas les np arrays

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host = '127.0.0.1', port = 5000)