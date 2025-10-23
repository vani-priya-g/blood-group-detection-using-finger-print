# blood-group-detection-using-finger-print
🩸 Blood Group Detection from Fingerprint Images using Deep Learning

Project Overview:
This project aims to predict a person’s blood group (A+, A−, B+, B−, AB+, AB−, O+, O−) directly from a fingerprint image using Convolutional Neural Networks (CNNs). By leveraging deep learning and computer vision, the system eliminates the need for traditional lab tests, providing a fast and non-invasive alternative for blood group estimation.

Key Features:

Upload a fingerprint image and receive instant blood group prediction.

Displays confidence score for each prediction, showing the model’s certainty.

Fully deployed as a Streamlit web application with a clean, user-friendly interface.

Trained on a custom fingerprint dataset labeled by blood groups.

Technologies Used:

Machine Learning & Deep Learning: TensorFlow, Keras, CNN

Web Development: Streamlit, Python

Data Handling & Processing: NumPy, OpenCV

Visualization & Analysis: Matplotlib, optional Seaborn

Deployment & Version Control: Git/GitHub

Project Workflow:

Data Collection:

Fingerprint images labeled with 8 major blood groups (A+, A−, B+, B−, AB+, AB−, O+, O−).

Model Training:

Convolutional Neural Network trained to recognize patterns in fingerprint ridges.

Model Saving:

Trained model stored in .h5 format for later inference.

Prediction App:

Streamlit app that preprocesses uploaded images, performs prediction, and shows the predicted blood group with confidence score.

Impact & Learning:

High accuracy in predicting blood groups from fingerprints.

Strengthened skills in deep learning, CNNs, image preprocessing, Python, and web deployment.

Demonstrated practical application of machine learning in non-invasive medical solutions.

Future Improvements:

Incorporate larger and more diverse datasets to improve accuracy.

Add Grad-CAM visualization to show which parts of the fingerprint the model focuses on.

Deploy on Streamlit Cloud or a web server for global access.

Integrate with mobile capture devices for real-time predictions.
