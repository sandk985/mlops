# flask api irish model 
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained model (ensure iris_model.pkl exists in the container)
model = joblib.load("iris_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]
    prediction = model.predict([features])[0]
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
