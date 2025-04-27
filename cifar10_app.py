import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 Dataset
@st.cache_data
def load_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize images
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()

# Class Names
class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Build a simple CNN model
@st.cache_resource
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()

# Streamlit App
st.title("🖼️ CIFAR-10 Image Classification App")
st.write("This app trains a CNN to classify images from the CIFAR-10 dataset!")

# Sidebar options
task = st.sidebar.selectbox("Select Task", ["Train Model", "Test Model on Random Image"])

if task == "Train Model":
    st.write("## Training Model...")
    if st.button('Start Training'):
        history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
        st.success("✅ Training Completed!")
        # Plot training history
        st.write("### Training History")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        ax[0].legend()
        ax[0].set_title('Accuracy')
        
        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Val Loss')
        ax[1].legend()
        ax[1].set_title('Loss')
        
        st.pyplot(fig)

if task == "Test Model on Random Image":
    idx = np.random.randint(0, len(X_test))
    img = X_test[idx]
    true_label = class_names[int(y_test[idx])]
    
    st.image(img, caption=f"True Label: {true_label}", width=150)
    
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_label = class_names[np.argmax(predictions)]
    
    st.write(f"### Predicted Label: **{predicted_label}**")

