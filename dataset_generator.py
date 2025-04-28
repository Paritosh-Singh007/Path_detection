import cv2
import os
import uuid

# Label-to-folder mapping
label_map = {
    'f': 'f',  # forward
    'l': 'l',  # left
    'r': 'r',  # right
    's': 's',  # stop
    'b': 'b',  # backward
    'd': 'destination'
}

url = 'https://10.21.82.163:8080/video'

# Create dataset folders if not already present
base_path = 'dataset'
os.makedirs(base_path, exist_ok=True)
for label in label_map.values():
    os.makedirs(os.path.join(base_path, label), exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(url)

print("Press f/l/r/s/b/d to capture an image for that class. Press q to quit.")

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ ERROR: Could not open webcam.")
    exit()

print("✅ Webcam opened successfully.")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Webcam - Press f/l/r/s/b", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif chr(key) in label_map:
        label = label_map[chr(key)]
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(base_path, label, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved: {filepath}")

cap.release()
cv2.destroyAllWindows()
