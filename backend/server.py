from flask import Flask, request, jsonify
from flask_cors import CORS
from processing import process_text
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        result = process_text(filepath)
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")  # Виведе помилку в консоль Flask
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
