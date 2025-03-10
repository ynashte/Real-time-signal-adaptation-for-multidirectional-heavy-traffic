from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('../YoloWeights/yolov8l.pt')
classNames = ['car', 'truck', 'bus', 'motorbike']

data = pd.read_csv("D://Mini Project Sem 5//traffic_signal_data.csv")
X = data[['noOfVehicles']].values
y = data['Time'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lasso1 = Lasso(alpha=0.5)
lasso1.fit(X, y)

def process_vehicle_and_time(frame):

    results = model(frame)
    vehicle_count = 0
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = box.conf[0]
        if cls < len(classNames) and classNames[cls] in classNames and conf >= 0.3:
            vehicle_count += 1

    predicted_time = int(lasso1.predict([[vehicle_count]])[0])

    annotated_frame = frame.copy()
    cv2.putText(annotated_frame, f'Vehicles Detected: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f'Predicted Time: {predicted_time:.2f} sec', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    max_width = 800
    height, width = annotated_frame.shape[:2]
    if width > max_width:
        aspect_ratio = max_width / float(width)
        new_height = int(height * aspect_ratio)
        resized_frame = cv2.resize(annotated_frame, (max_width, new_height))
    else:
        resized_frame = annotated_frame


    output_image_path = "static/output_image.jpg"
    cv2.imwrite(output_image_path, resized_frame)

    return vehicle_count, predicted_time, output_image_path


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:

        frame_data = request.json['frame']
        frame_data = frame_data.split(',')[1]
        frame_bytes = base64.b64decode(frame_data)


        frame = Image.open(BytesIO(frame_bytes))
        frame_np = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)


        vehicle_count, predicted_time, output_image_path = process_vehicle_and_time(frame_np)


        return jsonify({
            'vehicle_count': vehicle_count,
            'predicted_time': predicted_time,
            'output_image': '/' + output_image_path
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
