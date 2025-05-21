import cv2
import torch
import rawpy
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO  # Official YOLOv8 library from Ultralytics

# Load the YOLOv8 model (pre-trained on COCO dataset or a custom-trained model)
model = YOLO('yolov8n.pt')  # Use a pre-trained YOLOv8 model (yolov8n.pt, yolov8s.pt, etc.)


# Function to find the center of the ball and plot it
def find_and_plot_ball(image_path):
    # Read the NEF image using rawpy
    raw = rawpy.imread(image_path)
    rgb_image = raw.postprocess()

    # Convert to RGB for plotting with Matplotlib
    img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Perform object detection using YOLOv8
    results = model(rgb_image)  # Detect objects in the image

    # Results contains boxes, confidence, and class labels
    boxes = results[0].boxes  # Bounding box coordinates
    labels = results[0].names  # Object labels

    # Find the ball (replace 'ball' with the correct class if necessary)
    for box in boxes:
        label = labels[int(box.cls)]  # Get the label of the object detected
        if label == 'ball':  # Assuming 'ball' is the label (use appropriate label)
            x_center, y_center, width, height = box.xywh[0]  # Get the box coordinates

            # Convert to integer for visualization
            x_center, y_center = int(x_center), int(y_center)

            # Draw a circle on the ball's center
            cv2.circle(rgb_image, (x_center, y_center), 5, (0, 255, 0), -1)  # Green circle
            cv2.putText(rgb_image, f"Center: ({x_center}, {y_center})", (x_center + 10, y_center - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Show the image with the center of the ball marked
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.show()
            return x_center, y_center

    return None


# Path to your NEF image
image_path = 'test.NEF'  # Replace with your NEF file path

# Call the function to find and plot the ball center
center = find_and_plot_ball(image_path)

if center:
    print(f"Ball center at: {center}")
else:
    print("No ball detected in the image.")
