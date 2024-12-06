import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Path to your model and scaler
MODEL_PATH = os.getenv("MODEL_PATH", "diabetes.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

# Load the model
model = load_model(MODEL_PATH)

# Load the scaler (Assuming the scaler was saved using joblib)
scaler = joblib.load(SCALER_PATH)

@app.route('/predictions', methods=['POST'])
def predict():
    try:
        # Get the input features from the request
        data = request.get_json()

        # Validate input
        required_fields = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        if not all(field in data for field in required_fields):
            return jsonify({
                'success': False,
                'message': 'Missing required fields',
                'response_code': 400,
                'data': None
            }), 400

        # Extract the features
        features = np.array([[ 
            data['gender'],
            data['age'],
            data['hypertension'],
            data['heart_disease'],
            data['bmi'],
            data['HbA1c_level'],
            data['blood_glucose_level']
        ]])

        # Apply the scaler to the input features
        scaled_features = scaler.transform(features)

        # Predict with the model using the scaled features
        prediction = model.predict(scaled_features)

        # Get the predicted probability
        probability = prediction[0][0] * 100  # Asumming model gives a single probability

        # Determine the message based on the probability
        if probability <= 10:
            message = 'Risiko sangat rendah untuk terkena diabetes. Pola hidup sehat disarankan untuk mempertahankan kondisi ini.'
        elif probability <= 20:
            message = 'Risiko rendah. Terus pertahankan gaya hidup sehat, termasuk pola makan dan olahraga teratur.'
        elif probability <= 30:
            message = 'Risiko mulai meningkat. Disarankan untuk melakukan pemeriksaan lebih lanjut dan menjaga pola makan yang seimbang.'
        elif probability <= 40:
            message = 'Risiko sedang. Perlu konsultasi dengan dokter untuk mendapatkan rekomendasi terkait gaya hidup dan pengawasan kesehatan.'
        elif probability <= 50:
            message = 'Tanda-tanda awal diabetes terlihat. Anda perlu menjaga pola makan dan melakukan pemeriksaan rutin untuk menghindari komplikasi.'
        elif probability <= 60:
            message = 'Risiko diabetes semakin nyata. Disarankan untuk memonitor kadar gula darah secara rutin dan mengikuti saran medis.'
        elif probability <= 70:
            message = 'Terdapat gejala awal diabetes yang cukup signifikan. Pemeriksaan lebih lanjut dan perubahan gaya hidup sangat diperlukan.'
        elif probability <= 80:
            message = 'Risiko tinggi terkena diabetes kronis. Anda harus segera berkonsultasi dengan dokter untuk pengelolaan penyakit yang tepat.'
        elif probability <= 90:
            message = 'Kemungkinan besar diabetes sudah ada. Perawatan medis segera diperlukan untuk mencegah komplikasi lebih lanjut.'
        elif probability <= 100:
            message = 'Kondisi diabetes sudah sangat serius. Anda perlu segera mendapatkan pengobatan dan perawatan jangka panjang.'
        else:
            message = 'Data tidak valid. Silakan coba lagi.'

        # Prepare the response
        response = {
            'success': True,
            'message': 'Prediction made successfully',
            'response_code': 200,
            'data': {
                'message': message,
                'probability': probability,
                'prediction': int(prediction[0] > 0.5),
                'probabilities': prediction.tolist()
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'response_code': 500,
            'data': None
        }), 500

@app.route('/predictions', methods=['GET'])
def predictions():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    # app.run(debug=True)
