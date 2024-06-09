import streamlit as st
import numpy as np
import torch
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from ultralytics import YOLO

import os
import pandas as pd
from PIL import Image
import cv2

@st.cache(allow_output_mutation=True)
def get_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("rachmadiapri/TrOCRHandwriten-sirekap-v1", use_auth_token="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")   
    return processor,model

processor, model = get_model()

# Load the YOLOv8 model
#\sirekap-trocr-Deploy\model\best (2).pt
yolo_model_path = "best (2).pt"
yolo_model = YOLO(yolo_model_path)

# Function to perform object detection
def detect_objects(image):
    results = yolo_model.predict(image)  # Perform object detection
    results = results[0]
    if results:  # Check if results is not empty
        return results
    else:
        return []  # Return an empty list if no objects are detected

# Function to perform text recognition
def recognize_text(image, bounding_box):
    # Convert the bounding box to the format expected by TrOCR
    x1, y1, x2, y2 = bounding_box[0]
    crop = image[int(y1):int(y2), int(x1):int(x2)]

    # Preprocess the cropped image for TrOCR
    pixel_values = processor(images=crop, return_tensors="pt").pixel_values

    # Generate text predictions
    output = model.generate(pixel_values, max_length=50, early_stopping=True)
    recognized_text = processor.batch_decode(output, skip_special_tokens=True)[0]
    return recognized_text

# Streamlit app
def main():
    st.title("Object Detection and Text Recognition")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    button = st.button("Detect")

    if uploaded_file is not None and button:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', width=400)

        # Perform object detection
        results = detect_objects(image)

        # Iterate over detected bounding boxes
        recognized_texts = []
        if results:  # Check if results is not empty
            for i_label, box in enumerate(results.boxes):
                box = box.xyxy.tolist()  # Convert the bounding box to a list
                # Perform text recognition
                recognized_text = recognize_text(image, box)
                recognized_texts.append(recognized_text)
                # display image 
                x1, y1, x2, y2 = box[0]
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, results.names[i_label], (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        # Display the image with bounding boxes
        st.image(image, caption='Detected Objects', width=400)

        # Display the recognized text for each bounding box
        for i, text in enumerate(recognized_texts):
            st.write(f"Perolehan Suara Paslon {i+1}: {text}")

if __name__ == "__main__":
    main()
