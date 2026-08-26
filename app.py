import pandas as pd
import pickle

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model
model = pickle.load(open("gradient_boosting_claim_pipeline.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def prediction():

    data = request.form

    features = pd.DataFrame([{
        "age": float(data["age"]),
        "sex": data["sex"],
        "weight": float(data["weight"]),
        "bmi": float(data["bmi"]),
        "hereditary_diseases": data["hereditary_diseases"],
        "no_of_dependents": int(data["no_of_dependents"]),
        "smoker": int(data["smoker"]),
        "city": data["city"],
        "bloodpressure": float(data["bloodpressure"]),
        "diabetes": int(data["diabetes"]),
        "regular_ex": int(data["regular_ex"]),
        "job_title": data["job_title"]
    }])

    prediction = model.predict(features)

    return render_template(
        "index.html",
        prediction=round(float(prediction[0]), 2)
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0" , port=5000)