from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_filename = 'RandomForest.pkl'  # You can change this to other models if needed
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(data)[0]

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

