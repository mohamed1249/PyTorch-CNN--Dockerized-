# PyTorch CNN [Dockerized]
### Documentation for PyTorch Image Classification API

This document provides detailed instructions for setting up and running a Dockerized Flask API for image classification using a PyTorch model.

#### Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Setup Instructions](#setup-instructions)
4. [Directory Structure](#directory-structure)
5. [Flask API Code](#flask-api-code)
6. [Dockerfile](#dockerfile)
7. [Steps Taken](#steps-taken)
8. [Running the Application](#running-the-application)

### Project Overview

This project demonstrates how to create a Dockerized Flask API that uses a pre-trained PyTorch model to classify images of insects. Users can send a POST request with an image file, and the API will return the predicted class.

### Prerequisites

- Docker installed on your machine
- Basic knowledge of Python, Flask, and PyTorch

### Setup Instructions

1. **Clone the repository**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Download the dataset**
    - Download your dataset of insect images and place it in a directory named `selected_insects`.

3. **Create a virtual environment**
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4. **Install dependencies**
    ```sh
    pip install torch torchvision flask Pillow
    ```

5. **Train your model and save the state dictionary**
    - Use your training script to generate and save the model state dictionary as `model_state_dict.pt`.

6. **Write the Flask API code**: The `app.py` code is provided below.

### Directory Structure

```
project-directory/
│
├── app.py
├── Dockerfile
├── model_state_dict.pt
├── selected_insects/
│   ├── class1/
│   ├── class2/
│   ├── ...
└── requirements.txt
```

### Flask API Code

Below is the complete `app.py` code for the Flask API:

```python
from flask import Flask, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import io
from torchvision import transforms

label_set = os.listdir('selected_insects')
ns = 16 * 54 * 54

# Model Class
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(ns, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, len(label_set))

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, ns)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

# Load the state dictionary
state_dict = torch.load("model_state_dict.pt")

# Create the model
model = ConvolutionalNetwork()
model.load_state_dict(state_dict)
model.eval()

def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def decode_y(y, label_set):
    return label_set[int(y.argmax().cpu().numpy())]

app = Flask(__name__)

@app.route("/classify", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return {"error": "No image file provided"}, 400

    image_bytes = request.files["image"].read()
    image_tensor = preprocess_image(image_bytes)

    with torch.no_grad():
        prediction = model(image_tensor)

    label_set = os.listdir('selected_insects')
    label_set = [label.strip() for label in label_set]

    return {"class": decode_y(prediction, label_set)}

if __name__ == "__main__":
    app.run(debug=True)
```

### Dockerfile

Below is the complete `Dockerfile` for Dockerizing the Flask API:

```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir torch torchvision flask Pillow

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run flask when the container launches
CMD ["flask", "run"]
```

### Steps Taken

1. **Downloaded data**
    - Collected and organized insect images into a directory named `selected_insects`. You can find the dataset here: https://www.kaggle.com/datasets/shameinew/insect-images-with-scientific-names/data
2. **Created environment**
    - Set up a Python virtual environment for dependency management.
3. **Installed dependencies**
    - Installed necessary packages such as `torch`, `torchvision`, `flask`, and `Pillow`.
4. **Wrote code**
    - Developed the model architecture and training script.
    - Saved the trained model state dictionary as `model_state_dict.pt`.
    - Wrote the Flask API code in `app.py`.
5. **Generated model**
    - Trained the model and saved the state dictionary.
6. **Wrote Flask API code**
    - Developed the Flask API for handling image classification requests.
7. **Dockerization**
    - Created a Dockerfile to containerize the Flask API.

### Running the Application

1. **Build the Docker image**
    ```sh
    docker build -t pytorch-flask-app .
    ```

2. **Run the Docker container**
    ```sh
    docker run -p 5000:5000 pytorch-flask-app
    ```

3. **Send a classification request**
    - Use a tool like `curl` or Postman to send a POST request to `http://localhost:5000/classify` with an image file.

Example `curl` command:
```sh
curl -X POST -F 'image=@path_to_your_image.jpg' http://localhost:5000/classify
```

This should return a JSON response with the predicted class of the image.

By following this documentation, you should be able to set up, run, and test the PyTorch image classification API in a Docker container. If you encounter any issues or have questions, feel free to reach out for help.
