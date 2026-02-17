import cv2
import torch
import torch.nn as nn
import numpy as np
from statistics import mode


# =====================================================
# 1) Network Architecture (must match training)
# =====================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res1 = ResidualBlock(64)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res2 = ResidualBlock(128)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res3 = ResidualBlock(256)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.res1(self.conv1(x))
        x = self.res2(self.conv2(x))
        x = self.res3(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# =====================================================
# 2) Load Detection & Model
# =====================================================
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

model_path = "/Users/ly61/Desktop/cnn/model/model_cnn.pth"  # ← 替换成你自己的路径

model = FaceCNN()
state_dict = torch.load(model_path, map_location="cpu")

# Remove 'module.' if trained under DataParallel
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v

model.load_state_dict(new_state_dict)
model.eval()

print("Model loaded successfully\n")


# =====================================================
# 3) Emotion Labels
# =====================================================
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}

emotion_window = []
frame_window = 10


# =====================================================
# 4) Camera Loop
# =====================================================
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
print("Camera started — press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        if w < 10 or h < 10:
            continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), (84,255,159), 2)

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face.astype("float32") / 255.0
        face_tensor = torch.tensor(face).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = model(face_tensor).argmax(1).item()

        emotion = emotion_labels[pred]
        emotion_window.append(emotion)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)

        try:
            emotion_mode = mode(emotion_window)
        except:
            emotion_mode = emotion

        cv2.putText(frame, emotion_mode, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,0,255), 2)

    cv2.imshow("FER Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed!")
