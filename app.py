from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Memuat model yang telah dilatih
with open('DTRModel.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = [
        float(data['lebar_bangunan']),
        float(data['lebar_tanah']),
        float(data['jumlah_kamar_tidur']),
        float(data['jumlah_kamar_mandi']),
        float(data['jumlah_kapasitas_mobil_dan_garasi'])
    ]
    prediction = model.predict([features])[0]
    return jsonify({'prediction': f"Rp {prediction:,.2f}"})

@app.route('/result')
def result():
    # Di sini Anda bisa menambahkan logika untuk menampilkan hasil prediksi
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
