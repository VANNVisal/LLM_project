import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2

# ========== Config ==========
MODEL_PATH = r"C:\Users\BNC\Documents\ITC-project\plastic\small_model_cnn.pth"
NUM_CLASSES = 5  # <-- Must match the trained model
IMAGE_SIZE = 64
CLASS_NAMES = [f'class_{i}' for i in range(NUM_CLASSES)]  # Placeholder names

# ========== Model Definition ==========
class TrainedCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(TrainedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # (B,3,64,64) -> (B,16,64,64)
            nn.ReLU(),
            nn.MaxPool2d(2),                # -> (B,16,32,32)

            nn.Conv2d(16, 32, 3, padding=1),  # -> (B,32,32,32)
            nn.ReLU(),
            nn.MaxPool2d(2),                 # -> (B,32,16,16)

            nn.Conv2d(32, 64, 3, padding=1),  # This layer caused your error (features.6)
            nn.ReLU(),
            nn.MaxPool2d(2),                 # -> (B,64,8,8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # -> (B,64*8*8 = 4096)
            nn.Linear(64 * 8 * 8, 128),    # (4096 -> 128)
            nn.ReLU(),
            nn.Linear(128, num_classes)    # (128 -> num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ========== Load Model ==========
model = TrainedCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# ========== Define Transform ==========
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# ========== Real-time Camera ==========
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Resize and transform image
    img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = CLASS_NAMES[predicted.item()]

    # Draw prediction
    cv2.putText(frame, f"Prediction: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
