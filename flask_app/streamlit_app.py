import streamlit as st
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.cnn import CustomCNN
from utils.prep import get_pytorch_transforms
from models.cnn_tf import create_cnn_model

pytorch_model = CustomCNN(num_classes=4)
pytorch_model.load_state_dict(torch.load(os.path.join(project_root, 'model.pth'), map_location='cpu'))
pytorch_model.eval()

tf_model = tf.keras.models.load_model(os.path.join(project_root, 'model_tf.h5'))

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.markdown('<div class="header"><h1>Brain Tumor Diagnostics</h1></div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="welcome-message">Welcome to Brain Tumor Diagnostics – Your Health Matters</div>', unsafe_allow_html=True)
    st.markdown('<h2>Brain Tumor Classification</h2>', unsafe_allow_html=True)

    model_type = st.selectbox("Select the model", ["pytorch", "tensorflow"])
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=False)
        transform = get_pytorch_transforms()[1]
        if st.button("Predict"):
            try:
                if model_type == 'pytorch':
                    image_tensor = transform(image).unsqueeze(0)
                    with torch.no_grad():
                        output = pytorch_model(image_tensor)
                        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                        prediction = torch.argmax(output, dim=1).item()
                        confidence = probabilities[prediction] * 100
                else:
                    image_array = np.array(image.resize((224, 224))) / 255.0
                    image_array = image_array[np.newaxis, ...]
                    output = tf_model.predict(image_array)
                    probabilities = output[0]
                    prediction = np.argmax(output, axis=1)[0]
                    confidence = probabilities[prediction] * 100

                st.success(f"Prediction: {class_names[prediction]} (Confidence: {confidence:.2f}%)")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Developed by Ousmane TIENTA</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    st.cache_resource.clear()
