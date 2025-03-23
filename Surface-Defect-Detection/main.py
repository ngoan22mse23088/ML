import base64
import io
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageOps
import tensorflow as tf
from flask_cors import CORS

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your pre-trained model
model_path = '/Users/lechingoan/Downloads/Surface-Defect-Detection/surface_defect_detection_model.h5'
model = tf.keras.models.load_model(model_path, compile=False)  # Update the path to your model
class_names_file = '/Users/lechingoan/Downloads/Surface-Defect-Detection/labels.txt'
class_names = [line.strip() for line in open(class_names_file, "r")]

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = ImageOps.fit(img, (200, 200), Image.Resampling.LANCZOS)  # Example size
        img_array = np.asarray(img)
        img_array = (img_array.astype(np.float32) / 255.0)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict image
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        
        # Convert image array to base64 string
        img_pil = Image.fromarray((img_array[0] * 255).astype(np.uint8))
        img_buffer = io.BytesIO()
        img_pil.save(img_buffer, format='JPEG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        return jsonify({'predicted_class': predicted_class, 'image_base64': img_base64})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Định nghĩa route để xử lý yêu cầu GET và trả về "Hello, World!"
@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, World!'

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
