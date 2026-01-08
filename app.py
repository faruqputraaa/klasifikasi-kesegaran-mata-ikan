import os
import io
import base64
import torch
import torch.nn as nn
from flask import Flask, request, render_template, jsonify
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image

# =========================
# Flask App
# =========================
app = Flask(__name__)

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Model Configuration
# =========================
MODEL_PATH = "efficientnetv2_final_model_v2.pth"
NUM_CLASSES = 2

# Label HARUS sesuai training
class_names = ["Tidak Segar", "Segar"]

# =========================
# Load Model
# =========================
try:
    model = efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    print("✅ Model EfficientNetV2-S berhasil dimuat")

except Exception as e:
    print("❌ Gagal memuat model:", e)
    model = None

# =========================
# Image Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model tidak tersedia"}), 500

    if "file" not in request.files:
        return jsonify({"error": "File tidak ditemukan"}), 400

    file = request.files["file"]
    fish_type = request.form.get("fish_type", "Tidak diketahui")

    if file.filename == "":
        return jsonify({"error": "File kosong"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        prediction = class_names[pred_idx]
        confidence = probs[0][pred_idx].item() * 100

        # Encode image
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            "fish_type": fish_type,
            "prediction": prediction,
            "confidence": f"{confidence:.2f}",
            "image": img_base64
        })

    except Exception as e:
        print("❌ Error prediksi:", e)
        return jsonify({"error": "Gagal memproses gambar"}), 500


if __name__ == "__main__":
    if not os.path.exists("templates"):
        os.makedirs("templates")

    app.run(debug=True)
