import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torchvision import utils
from torchvision.datasets import ImageFolder
from torchsummary import summary
import torch.nn.functional as F
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model

class CNN_Retino(nn.Module):

    def __init__(self, params):
        super(CNN_Retino, self).__init__()

        Cin, Hin, Win = params["shape_in"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        # CNN Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h, w = findConv2dOutShape(Hin, Win, self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv2)
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv3)
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv4)

        # compute the flatten size
        self.num_flatten = h * w * 8 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X));
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)

transform = transforms.Compose(
    [
        transforms.Resize((255, 255)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_skin_cancer(image_data):
    st.write("Predicting using Skin Cancer model...")
    img = image.load_img(image_data, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    model = load_model("C:\\major\\Model.h5")
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    if predicted_class == 0:
        return "Benign"
    else:
        return "Malignant"

def predict_pneumonia(image):
    st.write("Predicting using Pneumonia model...")
    model = load_model("C:\\major\\Pneumonia_model.h5")
    img_width, img_height = 256, 256
    img = Image.open(image)
    img = img.convert("RGB")
    img = img.resize((img_width, img_height))
    img_array = np.array(img) / 256.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    if predicted_class == 0:
        return "Normal"
    else:
        return "Pneumonia"

def predict_diabetic_retinopathy(image):
    model = torch.load("C:\\major\\Retino_model.pt")
    model = model.to(device)
    model.eval()
    st.write("Predicting using Diabetic Retinopathy model...")
    image = Image.open(image).convert("RGB")  # Ensure image is in RGB format
    image_tensor = transform(image)  # Apply transformations
    image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

    # Make the prediction
    with torch.no_grad():  # Disable gradient calculation
        output = model(image_tensor)  # Forward pass through the model
        _, predicted = torch.max(output, 1)  # Get the predicted class

    # Map predicted class to label
    pred_label = "DR" if predicted.item() == 0 else "No_DR"

    return pred_label

# Streamlit App
st.title('Disease Predictor')

disease_option = st.selectbox(
    'Select the disease model for prediction:',
    ('Select an option', 'Skin Cancer', 'Pneumonia', 'Diabetic Retinopathy')
)

if disease_option != 'Select an option':
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        st.image(image_to_predict, caption='Uploaded Image', width=300)
        if disease_option == 'Skin Cancer':
            result = predict_skin_cancer(uploaded_file)
        elif disease_option == 'Pneumonia':
            result = predict_pneumonia(uploaded_file)
        elif disease_option == 'Diabetic Retinopathy':
            result = predict_diabetic_retinopathy(uploaded_file)
        else:
            result = "Invalid option selected."

        st.write(result)
else:
    st.write("Please select a disease model to proceed.")
