from flask import Flask, request, send_file, make_response
from flasgger import Swagger
from werkzeug.utils import secure_filename
import pickle as pkl
import pandas as pd
import os
from config import MODEL_PICKLE_PATH
import config

app = Flask(__name__)
Swagger(app)


@app.route("/predict", methods=['POST'])
def predict():
    """APP
    This is an UI to make predictions for the best model
    ---
    consumes:
      - multipart/form-data 
    parameters:
      - name: file_name
        in: formData
        type: file
        required: true
      - name: model_type
        in: formData
        enum: ['Regresson', 'Classification']
        type: string
        required: true
      - name: model_name
        in: formData
        enum: ['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6']
        type: string
        required: true
    
    responses:
      200:
        description: A valid predictions file in csv format
        "schema" : {
              "type" : "file"
              }
      400:
        description: Invalid request, missing params
        "schema" : {
              "type" : "string"
              }
    """

    file = request.files["file_name"]
    model = request.form.get("model_name")
    model_type = request.form.get("model_type")

    if model_type == 'Regresson':
      MODEL_PICKLE_PATH = os.path.join(config.MODEL_PICKLE_PATH, "Regresson")
    else:
      MODEL_PICKLE_PATH = os.path.join(config.MODEL_PICKLE_PATH, "Classification")

    file.save(secure_filename(file.filename))
    data = pd.read_csv(file.filename, header=0)

    try:
        with open(os.path.join(MODEL_PICKLE_PATH, model + ".pkl"), "rb") as model_file:
            model = pkl.load(model_file)
    except FileNotFoundError:
        response = make_response("SPECIFIED MODEL PICKLE NOT FOUND!!!", 400)
        return response

    model_prediction = model.predict(data)

    output_file_name = os.path.join(MODEL_PICKLE_PATH, "model_prediction.csv")
    preds = pd.DataFrame(model_prediction, columns=['target'])
    output = pd.concat([data, preds], axis=1)
    output.to_csv(output_file_name, index=False, header=True)

    return send_file(output_file_name, as_attachment=True)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)