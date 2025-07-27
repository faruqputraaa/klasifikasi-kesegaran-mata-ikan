# Gemini Deployment Process for Fish Freshness Classifier

This document outlines the process of creating a web-based deployment for the fish freshness classification model.

## 1. Understanding the Model

The first step was to analyze the existing Python script `improved_classification.py`. This revealed:

*   **Model Architecture:** The model is an `EfficientNet-V2-B0` from the `timm` library.
*   **Image Preprocessing:** The script uses a series of transforms, including resizing to 224x224, normalization, and data augmentation. For deployment, we only need the essential transforms: resize and normalize.
*   **Class Labels:** The model is trained to classify images into two categories: "Segar" (Fresh) and "Tidak Segar" (Not Fresh).

## 2. Creating the Deployment Files

Based on the analysis, the following files were created:

### `app.py`

This Python script uses the Flask framework to create a web server. Its key responsibilities are:

*   **Model Loading:** It loads the pre-trained `efficientnetv2_ikan.pt` model and sets it to evaluation mode.
*   **Web Routes:**
    *   `/`: Renders the main HTML page.
    *   `/predict`: Handles the image upload, preprocessing, and prediction.
*   **Image Handling:** It receives an uploaded image, applies the necessary transformations, and feeds it to the model.
*   **Prediction:** It returns the predicted class (either "Segar" or "Tidak Segar") and the model's confidence score as a JSON response.

### `templates/index.html`

This is the frontend of the application, built with Tailwind CSS for a modern and responsive design. It includes:

*   **Loading Screen:** A loading screen with the company logo is displayed while the model initializes.
*   **Camera and Upload Buttons:** Users can choose to either upload an image from their device or use their camera to capture a new one.
*   **Cropping Functionality:** It uses the `cropper.js` library to allow the user to crop the uploaded or captured image to a 1:1 aspect ratio. This ensures that the model receives a well-formatted input.
*   **Prediction Display:** It sends the cropped image to the server, receives the prediction, and displays the result, including the predicted class, confidence score, and the uploaded image.

## 3. Running the Application

To run the application, execute the following command in your terminal:

```bash
python app.py
```

This will start a local development server. You can then access the application in your web browser at `http://127.0.0.1:5000`.
