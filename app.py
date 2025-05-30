# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image
import os
import base64

app = Flask(__name__)
CORS(app)


trained_model_path = 'models/best.pt'

model = None
try:
    if os.path.exists(trained_model_path):
        model = YOLO(trained_model_path)
        print(f"YOLOv8 model successfully loaded: {trained_model_path}")
    else:
        print(f"Error: Trained model not found at: {trained_model_path}")
        print("Please ensure 'best.pt' model file is in the 'models/' directory.")
except Exception as e:
    print(f"Error loading model: {e}")


def remove_hairs(image_np):

    # Convert the image to grayscale for processing
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # 1. Black Hat Transformation:
    # Highlights dark features on a light background (like hairs on skin).
    # Kernel size should be adjusted based on hair thickness and image resolution.
    # A larger kernel might remove thicker hairs but could also remove fine details.
    kernel_size = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)) # Example: (10,10) for average hairs
    blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel_size)

    # 2. Gaussian Blurring:
    # Reduces noise in the blackhat image to improve contour detection.
    blurred_blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

    # 3. Binary Thresholding:
    # Creates a binary mask where potential hair regions are highlighted.
    # The threshold value (e.g., 10) depends on the contrast of hairs.
    ret, thresh = cv2.threshold(blurred_blackhat, 10, 255, cv2.THRESH_BINARY)

    # 4. Find Contours:
    # Detects the outlines of the potential hair regions.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # Example: (5,5) elliptical kernel
    mask = np.zeros(gray_image.shape[:2], np.uint8) # Create an empty mask
    for cnt in contours:
        cv2.drawContours(mask, [cnt], 0, 255, -1) # Draw contours onto the mask
    dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=1) # Dilate the mask

    cleaned_image = cv2.inpaint(image_np, dilated_mask, 3, cv2.INPAINT_TELEA) # 3: inpaint radius

    return cleaned_image

@app.route('/')
def index():
    """Returns a status message for the API's root endpoint."""
    return "Melanoma Detection API is running! Send a POST request to /predict with an image file."

@app.route('/predict', methods=['POST'])
def predict():
    """Processes incoming images and performs melanoma detection."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure the model is at the correct path and file exists.'}), 500

    # Check if 'file' part is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request. Please send an image file named "file".'}), 400

    file = request.files['file']
    # Check if a file was selected or if it's empty
    if file.filename == '':
        return jsonify({'error': 'No selected file. Please select an image file.'}), 400

    if file:
        try:
            # Read image bytes, convert to PIL Image, then to OpenCV (NumPy array)
            img_bytes = file.read()
            img_pil = Image.open(io.BytesIO(img_bytes))
            img_np = np.array(img_pil)
            if img_np.ndim == 3 and img_np.shape[2] == 3: # Convert RGB to BGR if needed
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


            processed_img_np = remove_hairs(img_np)
            # Create a copy for drawing, so original processed image is not altered for model input
            marked_image_np = processed_img_np.copy()
            print("Image processed for hair removal.")


            results = model(processed_img_np, conf=0.5, iou=0.45) # Use the processed image for prediction

            detections = []
            class_names = model.names # Class names from the trained model (from Roboflow data.yaml)

            for r in results:
                boxes = r.boxes # All detected bounding boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
                    confidence = float(box.conf[0])        # Confidence score
                    class_id = int(box.cls[0])             # Class ID

                    # Get class name from ID
                    class_name = class_names[class_id]

                    detections.append({
                        'class_name': class_name,
                        'confidence': round(confidence, 4), # Round confidence to 4 decimal places
                        'box': [x1, y1, x2, y2]
                    })


                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    # Set radius slightly larger than half of the max dimension of the box
                    radius = int(max(x2 - x1, y2 - y1) / 2) + 5 # Add 5 pixels for extra padding

                    color = (0, 0, 255) # Red color in BGR format
                    thickness = 2       # Line thickness

                    # Draw a circle at the center
                    cv2.circle(marked_image_np, (center_x, center_y), radius, color, thickness)

                    # Optionally, add class name and confidence score as text
                    text = f"{class_name}: {confidence:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_thickness = 2
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    # Position text above or below the circle
                    cv2.putText(marked_image_np, text, (x1, y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10),
                                font, font_scale, color, font_thickness, cv2.LINE_AA)


            encoded_img = None
            if marked_image_np is not None:
                # Convert BGR to RGB (as web typically displays RGB)
                marked_image_rgb = cv2.cvtColor(marked_image_np, cv2.COLOR_BGR2RGB)
                # Convert NumPy array to PIL Image object
                pil_img = Image.fromarray(marked_image_rgb)
                # Save the image to a byte buffer as JPEG
                byte_arr = io.BytesIO()
                pil_img.save(byte_arr, format='JPEG')
                # Convert to Base64
                encoded_img = base64.b64encode(byte_arr.getvalue()).decode('utf-8')

            response_data = {
                'status': 'success',
                'message': 'Melanoma detection performed successfully.',
                'predictions': detections,
                'marked_image_base64': encoded_img # Base64 encoded image is added here
            }

            return jsonify(response_data)

        except Exception as e:
            # Catch and return any errors during prediction or processing
            print(f"Error during prediction: {e}")
            return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)