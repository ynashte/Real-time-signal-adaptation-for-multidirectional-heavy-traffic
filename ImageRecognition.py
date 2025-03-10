from ultralytics import YOLO
import cv2

model = YOLO('../YoloWeights/yolov8l.pt')


classNames = ['car', 'truck', 'bus', 'motorbike']
image_path = "C:\\Users\91799\PycharmProjects\SemV\\vehicles.jpg"

results = model(image_path)
vehicle_count = 0

result = results[0]

for box in result.boxes:
    cls = int(box.cls[0])
    conf = box.conf[0]
    if cls < len(classNames) and classNames[cls] in classNames and conf >= 0.3:  # Filter out other classes
        vehicle_count += 1

annotated_image = result.orig_img
cv2.putText(annotated_image, f'Vehicles Detected: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Detected Vehicles', annotated_image)

print(f"Number of vehicles detected: {vehicle_count}")

cv2.waitKey(0)
cv2.destroyAllWindows()
