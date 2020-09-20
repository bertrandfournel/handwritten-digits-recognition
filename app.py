from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image, ImageOps
from joblib import load
import base64
from io import BytesIO


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

model = load('model.joblib')

@app.route('/get_result', methods=['POST'])
def get_result():
    data = request.data
    data_str = str(data)


    offset = data_str.index(',')+1

    img_bytes = base64.b64decode(data_str[offset:])
    img = Image.open(BytesIO(img_bytes))
    
    #On convertit l'image en niveaux de gris
    img_nb = img.convert("L")
        
    # On inverse les couleurs
    img_nb_invert = ImageOps.invert(img_nb)

    # On récupère un array des données de l'image
    arr = np.asarray(img_nb_invert)
    arr = arr.flatten()
    arr = arr.reshape(1,-1)

    # On fait la prédiction
    
    prediction = model.predict(arr)
    # print(model)
    return "I see "+str(prediction[0])



if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False,port="5000")