# 🐾 Animal Emotion Detection System
This project aims to detect and predict the emotions of animals (e.g., happy, sad, hungry) from images or videos. It also supports analyzing a group of animals and marking the emotions of each one. A notification system provides a summary of the detected emotions in the format:
"Animal Name + Emotion"

## 🚀 Features

### Emotion Detection for Animals:

Detect whether an animal is happy, sad, or hungry.

### Group Emotion Analysis:

Identify emotions for each animal in a group.

### Notification System:

Displays a pop-up message summarizing the emotions of all detected animals.

### Deep Learning Models:

Uses a trained CNN model (.h5 format) for emotion prediction and YOLOv8 for object detection.

## 📂 Dataset and Models

The system was trained on animal datasets with labeled emotions.

YOLOv8 was used for detecting animals in images/videos.

A custom-trained CNN model (.h5) predicts emotions for each detected animal.

## 🛠️ How It Works
### Input:

Upload an image or video containing one or more animals.

Detection and Emotion Prediction:

YOLOv8 detects animals and creates bounding boxes.

The CNN model predicts the emotion of each detected animal.

### Notification:

Displays a pop-up message with detected animals and their emotions.

## 🖼️ Example Output
### Single Animal:

Input: Image of a dog.

Output: Dog - Happy

Group of Animals:

Input: Image of a dog, cat, and bird.

Output: Dog - Hungry, Cat - Sad, Bird - Happy

## 📦 Installation
### Clone the repository:
git clone https://github.com/Hariarul/Animal-Emotion-Detection-
 
### Install dependencies:

pip install -r requirements.txt  

### Run app:
streamlit run Animal.py

## 🚨 Notification Example

### Pop-up Notification Example:
Single Animal:
Dog - Happy
Group of Animals:
Dog - Hungry, Cat - Sad, Bird - Happy
## 🏗️ Built With
TensorFlow: For training the CNN model.
YOLOv8: For object detection.

## 🎯 Results
### 📷 Example Results for Single Animal:
Input Image:
Output:
Dog - Happy
### 📷 Example Results for Group of Animals:
Input Image:
Output:
Dog - Hungry, Cat - Sad, Bird - Happy
### 🔔 Pop-up Notification Example:
For Single Animal:
Dog - Happy
For Group of Animals:
Dog - Hungry, Cat - Sad, Bird - Happy

## Python: Core programming language.
