import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# Load YOLO model
model = YOLO('../YoloWeights/yolov8l.pt')
classNames = ['car', 'truck', 'bus', 'motorbike']


# Lasso Regression Model
def load_lasso_model():
    # Load dataset for Lasso Regression
    data = pd.read_csv("/TrafficData.csv")  # Replace with actual path
    X = data[['noOfVehicles']].values
    y = data['Time'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    return lasso


lasso_model = load_lasso_model()


# Function to detect vehicles using YOLO
def detect_vehicles(image_path):
    results = model(image_path)
    vehicle_count = 0
    result = results[0]

    # Iterate over results to count vehicles
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = box.conf[0]
        if cls < len(classNames) and classNames[cls] in classNames and conf >= 0.3:
            vehicle_count += 1

    annotated_image = result.orig_img
    cv2.putText(annotated_image, f'Vehicles Detected: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    return vehicle_count, annotated_image


# Function to predict traffic light timings using Lasso Regression
def predict_traffic_time(no_of_vehicles):
    predicted_time = lasso_model.predict([[no_of_vehicles]])
    return predicted_time[0]


# Function to open an image file
def open_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.png")])
    if file_path:
        vehicle_count, annotated_image = detect_vehicles(file_path)

        # Convert the OpenCV image to Pillow format
        annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        annotated_image_tk = ImageTk.PhotoImage(annotated_image_pil)

        # Update the label with the image
        image_label.config(image=annotated_image_tk)
        image_label.image = annotated_image_tk  # Keep reference to avoid garbage collection

        # Use the detected vehicle count for traffic light prediction
        predicted_time = predict_traffic_time(vehicle_count)
        messagebox.showinfo("Vehicle Detection",
                            f"Vehicles detected: {vehicle_count}\nPredicted Traffic Light Timing: {predicted_time:.2f} seconds")


# Create the main window
root = tk.Tk()
root.title("Traffic Management System")

# Set the window size and background color
root.geometry("900x700")
root.config(bg="#E9F1F7")  # Light background color

# Create the top navigation bar
navbar = tk.Frame(root, bg="#4CAF50", height=50)
navbar.pack(fill="x")

# Add a title label in the navbar
title_label = tk.Label(navbar, text="Traffic Management System", font=("Arial", 20, "bold"), fg="white", bg="#4CAF50")
title_label.pack(pady=10)

# Create the main content area
content_frame = tk.Frame(root, bg="#E9F1F7", pady=20)
content_frame.pack(fill="both", expand=True)

# Card for Image and Results
card_frame = tk.Frame(content_frame, bg="white", width=800, height=500, padx=20, pady=20, relief="solid", bd=2)
card_frame.pack(pady=20)

# Create a label for showing images
image_label = tk.Label(card_frame, bg="white")
image_label.pack()

# Add a button to open an image and detect vehicles
open_image_button = tk.Button(root, text="Detect Vehicles", width=20, height=2, bg="#4CAF50", fg="white",
                              font=("Arial", 14, "bold"),
                              relief="solid", bd=2, command=open_image)
open_image_button.pack(pady=20)

# Add a footer with information about the app
footer_frame = tk.Frame(root, bg="#4CAF50", height=50)
footer_frame.pack(side="bottom", fill="x")
footer_label = tk.Label(footer_frame, text="Traffic light timing based on vehicle count.", font=("Arial", 12),
                        fg="white", bg="#4CAF50")
footer_label.pack(pady=10)

# Run the main loop
root.mainloop()
