import os
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)

# --- Model Loading ---
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model
model_path = 'efficientnetv2_ikan.pt'
num_classes = 2

try:
    # Load the model architecture using timm
    model = timm.create_model('tf_efficientnetv2_b0', pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    class_names = ['Segar', 'Tidak Segar'] 
    print("Model loaded successfully!")

except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None

# --- Image Transformations ---
# Transformation pipeline for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Read the image file
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Apply transformations
        img_t = transform(image)
        batch_t = torch.unsqueeze(img_t, 0).to(device)

        # Make a prediction
        with torch.no_grad():
            output = model(batch_t)
            _, predicted_idx = torch.max(output, 1)
            prediction = class_names[predicted_idx.item()]
            
            # Get probability scores
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            confidence = probabilities[predicted_idx.item()].item() * 100

        # Prepare the image for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'prediction': prediction,
            'confidence': f'{confidence:.2f}',
            'image': img_str
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error processing the image. Please try again.'}), 500

if __name__ == '__main__':
    # Create a templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Note: In a production environment, use a proper WSGI server like Gunicorn or uWSGI
    app.run(debug=True)
