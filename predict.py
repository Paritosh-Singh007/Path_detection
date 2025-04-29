import cv2
import torch
import serial
import time
import numpy as np
from PIL import Image
from torchvision import transforms
from model import CNNModel  # Your CNN class

# Load the trained model
model = CNNModel()
model.load_state_dict(torch.load('cnn_model.pth', map_location=torch.device('cpu')))
model.eval()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Bluetooth Serial (Arduino)
ser = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)  # Give Arduino time to reset

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

labels = ['f', 'l', 'r', 's', 'b', 'destination']
url = 'http://10.21.82.163:8080/video'
cap = cv2.VideoCapture(url)

def get_lidar_distance():
    ser.write('d'.encode())
    time.sleep(0.05)
    if ser.in_waiting:
        try:
            dist = ser.readline().decode().strip()
            return int(dist)
        except:
            return -1
    return -1

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Frame not captured!")
        continue
    else:
        print("✅ Frame captured")

    # Check LiDAR
    distance = get_lidar_distance()
    print(f"📏 Distance from LiDAR: {distance} cm")

    if 0 < distance <= 15:
        print("🛑 Obstacle/Destination within 15 cm!")
        ser.write('s'.encode())
        continue  # Skip CNN prediction if obstacle too close

    # CNN prediction
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    image = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = labels[predicted.item()]

    # Handle destination
    if predicted_class == 'destination':
        print("✨ Destination reached!")
        ser.write('s'.encode())
    else:
        ser.write(predicted_class.encode())

    # Show frame
    cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('Robot Vision', frame)
    print("👀 Showing frame in window...")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
