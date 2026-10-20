from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Memuat model regresi linear yang telah dilatih
model = joblib.load('model_insurance.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    bmi_input = None

    if request.method == 'POST':
        try:
            # Mengambil input BMI dari form
            bmi_input = float(request.form['bmi'])
            
            # Melakukan prediksi menggunakan model
            prediction = model.predict(np.array([[bmi_input]]))
            prediction_value = max(0, prediction[0]) # Memastikan hasil tidak negatif
            
            prediction_text = f"${prediction_value:,.2f}"
        except ValueError:
            prediction_text = "Input tidak valid. Silakan masukkan angka."

    return render_template('index.html', prediction_text=prediction_text, bmi_input=bmi_input)

if __name__ == '__main__':
    app.run(debug=True)