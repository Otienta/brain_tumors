import streamlit as st
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# Add the root directory to the search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.cnn import CustomCNN
from utils.prep import get_pytorch_transforms
from models.cnn_tf import create_cnn_model

# Load the models
pytorch_model = CustomCNN(num_classes=4)
pytorch_model.load_state_dict(torch.load(os.path.join(project_root, 'model.pth'), map_location='cpu'))
pytorch_model.eval()

tf_model = tf.keras.models.load_model(os.path.join(project_root, 'model_tf.h5'))

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# CSS personnalisé pour le style médical
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 100vh;
        color: #333333;
    }

    /* En-tête */
    .header {
        background-color: #ffffff;
        width: 100%;
        padding: 15px 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
    }

    .header h1 {
        color: #003087; /* Bleu foncé médical */
        font-size: 28px;
        font-weight: 700;
        font-family: 'Roboto', sans-serif;
    }

    /* Section principale */
    .custom-container {
        background: #ffffff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        max-width: 600px;
        width: 90%;
        text-align: center;
        margin: 20px auto;
        border: 3px solid #1e90ff; /* Bordure bleue */
    }

    .welcome-message {
        color: #003087; /* Bleu foncé médical */
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 20px;
        font-family: 'Roboto', sans-serif;
    }

    /* Style pour le titre */
    h2 {
        color: #003087;
        font-size: 22px;
        font-weight: 400;
        margin-bottom: 20px;
        font-family: 'Roboto', sans-serif;
    }

    /* Style pour les boutons */
    .stButton>button {
        background-color: #1e90ff; /* Bleu hospitalier */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 12px;
        font-weight: 700;
        font-family: 'Roboto', sans-serif;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #1c86ee;
        box-shadow: 0 5px 15px rgba(30, 144, 255, 0.4);
    }

    /* Style pour le selectbox */
    .stSelectbox select {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 12px;
        font-family: 'Roboto', sans-serif;
    }

    /* Style pour l'image */
    .stImage img {
        max-width: 300px;
        max-height: 300px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }

    .stImage img:hover {
        transform: scale(1.05);
    }

    /* Style pour les messages de succès et d'erreur */
    .stSuccess {
        color: #333333;
        font-family: 'Roboto', sans-serif;
    }

    .stError {
        color: #e74c3c;
        font-family: 'Roboto', sans-serif;
    }

    /* Pied de page */
    .footer {
        background-color: #1e90ff;
        width: 100%;
        text-align: center;
        padding: 15px 0;
        color: #ffffff;
        font-size: 14px;
        font-weight: 400;
        font-family: 'Roboto', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# En-tête
st.markdown('<div class="header"><h1>Brain Tumor Diagnostics</h1></div>', unsafe_allow_html=True)

# Section principale
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

# Pied de page
st.markdown('<div class="footer">Developed by Ousmane TIENTA</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    st.cache_resource.clear()