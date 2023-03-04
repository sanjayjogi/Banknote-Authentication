from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "welcome"


@app.route('/predict', methods=['GET'])
def predic_note_authenticattion():
    """Authenticating Bank Notes
    ---
    parameters:
        - name: variance
          in: query
          type: number
          require: true
        - name: skewness
          in: query
          type: number
          require: true
        - name: curtosis
          in: query
          type: number
          require: true
        - name: entropy
          in: query
          type: number
          require: true
    responses:
          200:
              description: the output values
    """

    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return 'predicted values:' + str(prediction)


@app.route('/predict_file', methods=['POST'])
def predict_note_file():
    """Authenticating bank notes
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true    

    responses:
          200:
              description: The output values 
    """
    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction = classifier.predict(df_test)

    return str(list(prediction))


if __name__ == '__main__':
    app.run(debug=True)
