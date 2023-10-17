import pickle
from flask import Flask, request, jsonify

model_file = "model2.bin"
vectorizer_file = "dv.bin"


with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)

with open(vectorizer_file, "rb") as f_in:
    dv = pickle.load(f_in)

app = Flask("credit")


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    credit = y_pred >= 0.5

    result = {"credit_probability": float(y_pred), "credit": bool(credit)}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
