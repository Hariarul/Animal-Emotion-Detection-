import cv2
import numpy as np
import streamlit as st
import cvzone
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import tempfile

# Load YOLO model and emotion models
try:
    yolo_model = YOLO("yolov8n-seg.pt")  # Replace with your YOLO model file path
    dog_emotion_model = load_model("dog_emotions.h5")
    cat_emotion_model = load_model("cat_emotions.h5")
except Exception as e:
    st.error(f"Error loading models: {e}")

# Class names for YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

def predict_emotion(animal_roi, model, animal_class):
    """Predict emotion for a given animal ROI."""
    resized_roi = cv2.resize(animal_roi, (150, 150)) / 255.0
    predictions = model.predict(np.expand_dims(resized_roi, axis=0))

    # Debug the predictions
    print(f"Predictions: {predictions}")
    print(f"Predictions shape: {predictions.shape}")

    # Define emotion labels based on the animal class
    if animal_class == "dog":
        emotion_labels = ["angry", "happy", "relaxed", "sad"]
    elif animal_class == "cat":
        emotion_labels = ["attentive", "relaxed"]
    else:
        emotion_labels = ["Unknown"]

    # Get the emotion prediction
    if predictions.shape[-1] == len(emotion_labels):
        emotion = emotion_labels[np.argmax(predictions)]
    else:
        emotion = "Unknown"  # If predictions don't match the expected size
    
    return emotion

def resize_frame(frame, target_width=1280):
    """Resize frame while preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale = target_width / w
    new_height = int(h * scale)
    resized_frame = cv2.resize(frame, (target_width, new_height))
    return resized_frame

def draw_label(frame, animal_name, emotion, x1, y1, x2, y2, color):
    """Draw animal name and emotion inside bounding box."""
    text1 = f"{animal_name}"
    text2 = f"{emotion}"

    text1_size, _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text2_size, _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

    text1_w, text1_h = text1_size
    text2_w, text2_h = text2_size

    text_y = y1 - 10
    cv2.rectangle(frame, (x1, text_y), (x1 + max(text1_w, text2_w) + 5, text_y + text1_h + text2_h + 5), color, -1)

    cvzone.putTextRect(frame, text1, (x1, text_y + text1_h), colorR=(255, 0, 255), colorT=(255, 255, 255), scale=3, thickness=3)
    cvzone.putTextRect(frame, text2, (x1, text_y + text1_h + 30 + text2_h), colorR=(255, 0, 255), colorT=(255, 255, 255), scale=3, thickness=3)

# Streamlit App UI
st.title("Animal Emotion Detection System")
st.write("Upload a video or image to detect animal emotions.")

# File uploader
uploaded_file = st.file_uploader("Choose a video or image", type=["mp4", "jpg", "png"])

# Process uploaded video or image
if uploaded_file is not None:
    # Check if the uploaded file is an image or a video
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        # If it's an image, we process it directly
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        frame_resized = resize_frame(image, target_width=1280)

        # Run YOLO object detection
        results = yolo_model(frame_resized, stream=True)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = box.conf
                if conf < 0.5:
                    continue
                currentClass = classNames[cls]
                if currentClass in ["dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]:
                    roi = frame_resized[y1:y2, x1:x2]
                    emotion_model = dog_emotion_model if currentClass == "dog" else cat_emotion_model
                    emotion = predict_emotion(roi, emotion_model, currentClass)
                    color = (0, 255, 0) if currentClass == "dog" else (255, 0, 0)
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                    draw_label(frame_resized, currentClass.capitalize(), emotion, x1, y1, x2, y2, color)

        # Show processed image
        st.image(frame_resized, channels="BGR", caption="Processed Image", use_column_width=True)

    elif uploaded_file.type == "video/mp4":
        # If it's a video, process it frame by frame
        with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
            temp_video_file.write(uploaded_file.read())
            temp_video_path = temp_video_file.name

        cap = cv2.VideoCapture(temp_video_path)
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = resize_frame(frame, target_width=1280)
            results = yolo_model(frame_resized, stream=True)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls)
                    conf = box.conf
                    if conf < 0.5:
                        continue
                    currentClass = classNames[cls]
                    if currentClass in ["dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]:
                        roi = frame_resized[y1:y2, x1:x2]
                        emotion_model = dog_emotion_model if currentClass == "dog" else cat_emotion_model
                        emotion = predict_emotion(roi, emotion_model, currentClass)
                        color = (0, 255, 0) if currentClass == "dog" else (255, 0, 0)
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                        draw_label(frame_resized, currentClass.capitalize(), emotion, x1, y1, x2, y2, color)

            frame_placeholder.image(frame_resized, channels="BGR", caption="Processing...", use_column_width=True)
    else:
        st.write("Please upload a valid image or video file.")
else:
    st.write("Please upload a video or image to get started.")
