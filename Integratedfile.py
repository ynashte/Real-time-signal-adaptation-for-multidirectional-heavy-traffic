from ultralytics import YOLO
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

model = YOLO('../YoloWeights/yolov8l.pt')


classNames = ['car', 'truck', 'bus', 'motorbike']

image_path = "C:\\Users\91799\PycharmProjects\SemV\\vehicles.jpg"

results = model(image_path)

vehicle_count = 0

result = results[0]

for box in result.boxes:
    cls = int(box.cls[0])
    conf = box.conf[0]
    if cls < len(classNames) and classNames[cls] in classNames and conf >= 0.3:
        vehicle_count += 1

annotated_image = result.orig_img

cv2.putText(annotated_image, f'Vehicles Detected: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

data = pd.read_csv("/TrafficData.csv")  # Replace with your dataset path


X = data[['noOfVehicles']].values
y = data['Time'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

predicted_time = lasso.predict([[vehicle_count]])

print(f"Predicted Time for {vehicle_count} vehicles is: {predicted_time[0]} seconds")


cv2.putText(annotated_image, f'Predicted Time: {predicted_time[0]:.2f} sec', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


cv2.imshow('Final Result', annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
