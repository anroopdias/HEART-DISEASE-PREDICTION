from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


# Load the trained Random Forest model
model_path = 'heartdiseasemodelbest111.joblib'


with open(model_path, 'rb') as file:
    model = joblib.load(file)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        features = [float(request.form[f"feature{i}"]) for i in range(1, 14)]
        
        # Create a Pandas DataFrame from the input data
        input_data = pd.DataFrame([features], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

        # Make a prediction
        prediction = model.predict(input_data)

        return render_template('index.html', prediction=prediction[0])
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=8080)
