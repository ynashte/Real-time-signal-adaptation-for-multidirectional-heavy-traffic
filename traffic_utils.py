from ultralytics import YOLO
import cv2
import os
import pandas as pd
from sklearn.linear_model import Lasso

# Load YOLO model and Lasso regression model
model = YOLO('../YoloWeights/yolov8l.pt')
lasso_model = Lasso(alpha=0.1)

# Load dataset and train Lasso model
data = pd.read_csv("/TrafficData.csv")
X = data[['noOfVehicles']].values
y = data['Time'].values
lasso_model.fit(X, y)

classNames = ['car', 'truck', 'bus', 'motorbike']

def detect_vehicles(frame):
    """ Use YOLO to detect vehicles and count them. """
    results = model(frame)
    vehicle_count = 0

    # Process YOLO detection results
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = box.conf[0]
        if cls < len(classNames) and classNames[cls] in classNames and conf >= 0.3:
            vehicle_count += 1

    # Annotate frame with vehicle count and save
    annotated_frame = results[0].orig_img
    cv2.putText(annotated_frame, f'Vehicles Detected: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    annotated_path = os.path.join('static/uploads', 'annotated_image.jpg')
    cv2.imwrite(annotated_path, annotated_frame)

    return vehicle_count, annotated_path

def predict_timing(vehicle_count):
    """ Use Lasso model to predict traffic light timing based on vehicle count. """
    return lasso_model.predict([[vehicle_count]])[0]
