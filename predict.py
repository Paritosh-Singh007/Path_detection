'''import cv2
import torch
import serial
import numpy as np
from PIL import Image
from torchvision import transforms
from model import CNNModel  # Make sure this is the CNN model you defined

# Load the trained model
model = CNNModel()
model.load_state_dict(torch.load('cnn_model.pth'))
model.eval()

# Set the device (CPU or GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize Bluetooth serial communication (make sure to use the correct port)
ser = serial.Serial('COM5', 9600, timeout=1)

# Define the transformation (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust if needed
])

# Mapping output of model to commands
labels = ['f', 'l', 'r', 's', 'b', 'destination']  # Added 'destination' label

# Open IP webcam stream
url = 'http://10.21.82.163:8080/video'  # Note: use http not https
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Frame not captured! Check URL or connection.")
        continue
    else:
        print("✅ Frame captured")

    # Convert to RGB and then to PIL image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Apply transform and move to device
    image = transform(pil_image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = labels[predicted.item()]

    # Send command based on prediction
    if predicted_class == 'destination':
        print("✨ Destination reached!")
        ser.write('s'.encode())  # Stop the bot
    else:
        ser.write(predicted_class.encode())  # Send the predicted command normally

    # Display frame with prediction
    cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('Robot Vision', frame)
    print("👀 Showing frame in window...")

    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
ser.close()'''


import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import CNNModel  # Make sure this is the CNN model you defined

# Load the trained model
model = CNNModel(num_classes=6)  # 🔵 Important: specify correct number of classes
model.load_state_dict(torch.load('cnn_model.pth', map_location='cpu'))  # load safely even if GPU model
model.eval()

# Set the device (CPU or GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ⚡ Commented Arduino Serial Communication for OpenCV Testing
# import serial
# ser = serial.Serial('COM5', 9600, timeout=1)

# Define the transformation (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Use same normalization as training
])

# Mapping output of model to commands
labels = ['f', 'l', 'r', 's', 'b', 'destination']

# Open IP webcam stream
url = 'http://10.21.82.163:8080/video'  # Make sure your IP webcam app is running
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Frame not captured! Check URL or connection.")
        continue
    else:
        print("✅ Frame captured")

    # Convert to RGB and then to PIL image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Apply transform and move to device
    image = transform(pil_image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = labels[predicted.item()]

    # Print the predicted command instead of sending to Arduino
    if predicted_class == 'destination':
        print("✨ Destination reached! (Would send 's')")
        # ser.write('s'.encode())  # 🚫 Commented for OpenCV test
    else:
        print(f"➡️ Command to send: {predicted_class}")
        # ser.write(predicted_class.encode())  # 🚫 Commented for OpenCV test

    # Display frame with prediction
    cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('Robot Vision', frame)
    print("👀 Showing frame in window...")

    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
# ser.close()  # 🚫 Commented for OpenCV test
