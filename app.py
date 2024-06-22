from flask import Flask, request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
import io

label_set = os.listdir('selected_insects')
ns = 16* 54* 54

# Model Class
class ConvolutionalNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 3, 1)
    self.conv2 = nn.Conv2d(6,16,3,1)
    # Fully Connected Layer
    self.fc1 = nn.Linear(ns, 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, len(label_set))

  def forward(self, X):
    X = F.relu(self.conv1(X))
    X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2
    # Second Pass
    X = F.relu(self.conv2(X))
    X = F.max_pool2d(X,2,2) # 2x2 kernal and stride 2

    # Re-View to flatten it out
    X = X.view(-1, ns) # negative one so that we can vary the batch size

    # Fully Connected Layers
    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = self.fc3(X)
    return F.log_softmax(X, dim=1)

# Load the state dictionary
state_dict = torch.load("model_state_dict.pt")

# Create the model (if necessary)
model = ConvolutionalNetwork()

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set to evaluation mode (optional)
model.eval()

# Define some helper functions
def preprocess_image(image_bytes):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to expected size
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)  # Add a batch dimension

def decode_y(y, label_set):
    return label_set[int(y.argmax().cpu().numpy())]

# Initialize Flask app
app = Flask(__name__)

# Define API endpoint for image classification
@app.route("/classify", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return {"error": "No image file provided"}, 400

    image_bytes = request.files["image"].read()
    image_tensor = preprocess_image(image_bytes)

    with torch.no_grad():
        prediction = model(image_tensor)

    # Load the class labels from a separate file (assuming it exists)
    label_set = os.listdir('insects')
    label_set = [label.strip() for label in label_set]

    return {"class": decode_y(prediction, label_set)}

if __name__ == "__main__":
    app.run(debug=True)
