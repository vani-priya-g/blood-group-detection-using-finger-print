import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

st.title("🩸 Blood Group Detection from Fingerprint")
st.write("Upload a fingerprint image to predict the blood group with confidence score.")

# Load trained model
model = load_model('finger_print/blood_group_model.h5')
classes = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# File uploader
uploaded_file = st.file_uploader("Choose a fingerprint image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    image = load_img(uploaded_file, target_size=(128, 128))
    st.image(image, caption='Uploaded Fingerprint', use_container_width=True)
    
    img_array = img_to_array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    result = classes[class_idx]
    confidence = round(float(np.max(prediction))*100, 2)  # confidence in %

    # Display results
    st.success(f"Predicted Blood Group: **{result}**")
    st.info(f"Confidence Score: **{confidence}%**")
