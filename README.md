# Automated-Face-Mask-Detection-System-using-Open-CV-and-Keras

Problem Statement

Theme: Healthcare

Problem Addressed:
The COVID-19 pandemic has necessitated the widespread use of face masks to mitigate virus transmission. Despite mandates, compliance monitoring is challenging, especially in high-traffic areas such as public transport, malls, industries, worship places, theaters, and other gathering spots. This lack of monitoring systems has led to an increase in COVID-19 cases, endangering public health.

Project Overview:

To address this issue, we developed a Convolutional Neural Network (CNN) model to detect the presence of face masks on individuals. Utilizing TensorFlow, OpenCV, and deep learning techniques, our system can identify whether a person is wearing a face mask in real-time on images and video streams. This project aims to enhance public safety by ensuring mask compliance in crowded spaces.

Technologies Used:

Google Colab: Used to build and train the CNN model due to its provision of free GPU resources.
Kaggle: Source of the dataset used for training and testing the model.
Keras: High-level neural network API utilized to simplify the model building and training process.
Image Visualization and Data Augmentation: Employed to analyze data structure and extract image features.
Convolutional Neural Network (CNN): Core technology for classifying images into "mask" and "no mask" categories.
OpenCV: Used for real-time image processing and detection.
Implementation:

Data Collection: Downloaded dataset from Kaggle.
Model Training:
Built and trained a CNN model using Google Colab.
Applied data augmentation techniques to improve model generalization.
Model Testing:
Tested the trained model on new static images and video streams to evaluate its performance.
Real-Time Prediction: Integrated the model with OpenCV to enable real-time face mask detection in videos.
Languages Used:

Python
Benefits to Society:

The project aims to promote public health by ensuring individuals comply with face mask mandates in crowded public places. By installing this system in locations such as malls, bus stands, and other public areas, it can help reduce the spread of COVID-19 and potentially save lives.

Future Scope:

As the need for face masks may continue due to future pandemics, rising air pollution, and other health concerns, this face mask detection system will remain relevant. Future enhancements could include:

Improved accuracy and speed of detection.
Integration with other safety monitoring systems.
Expansion to detect other types of protective equipment.
Model Architecture:

Input Layer: Receives images for processing.
Convolutional Layers: Apply convolution operations to extract features.
Pooling Layers: Reduce dimensionality while retaining important features.
Flattening Layer: Converts pooled feature maps into a single long feature vector.
Fully Connected Layers: Perform classification based on the extracted features.
Training Process:

Training Set: Used to learn the model features.
Validation Set: Ensures the model does not overfit the training data.
Test Set: Evaluates the model performance after training.
Graphs for Loss and Accuracy:

Loss Graph: Indicates model overfitting if training loss decreases while validation loss increases.
Accuracy Graph: Shows the accuracy of the model over training epochs.
CNN on Sample Human Face Dataset:

Feature Learning: The second convolution layer learns complex parts of the face.
Edge Detection: The model uses edges and angles to recognize mask presence.
Activation Functions: Apply non-linear transformations to learn complex patterns.
By implementing this automated face mask detection system, we hope to contribute to public health safety and assist in the fight against COVID-19 and future pandemics.
