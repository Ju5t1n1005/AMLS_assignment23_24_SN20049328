# Introduce the necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

def run_task_B():

    # Setting random seeds to ensure reproducible results
    np.random.seed(42)
    tf.random.set_seed(42)

    # dataset path, please replace with the actual path of your PathMNIST dataset
    file_path = 'Datasets/pathmnist.npz'

    # Load the .npz file
    data = np.load(file_path)
    train_images = data['train_images']
    val_images = data['val_images']
    test_images = data['test_images']
    train_labels = data['train_labels']
    val_labels = data['val_labels']
    test_labels = data['test_labels']

    # Pre-process data: Normalise image data to 0-1 range to optimise training process
    train_images = train_images.astype('float32') / 255
    val_images = val_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # Convert tags to one-hot coding for easy categorisation
    train_labels = to_categorical(train_labels, 9)
    val_labels = to_categorical(val_labels, 9)
    test_labels = to_categorical(test_labels, 9)

    # Building CNN models: using a VGG-style network structure
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 3)),     # Input layer: receives image data of 28x28x3 size
        Conv2D(32, (3, 3), padding='same', activation='relu'),                              # The second convolutional layer
        MaxPooling2D(2, 2),                                                                 # Pooled layers, reduced feature map size, reduced number of parameters
        Dropout(0.3),               # Increase Dropout rate to reduce overfitting 
        Conv2D(64, (3, 3), padding='same', activation='relu'),                              # More convolutional and pooling layers
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(9, activation='softmax')                                      # Output layer: use softmax activation function to output probabilities for 9 categories
    ])

    # Set early stop to prevent overfitting, stop training if no lift in 3 iterations
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # Compilation models: using the adam optimiser and multiclass cross-entropy loss functions
    model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_images, 
        train_labels, 
        batch_size=64, 
        epochs=30, 
        validation_data=(val_images, val_labels), 
        callbacks=[early_stopping]
    )

    # Plot the training and validation loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    # Test and evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")

    # Generate a confusion matrix and classification report
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)

    # Plot the confusion matrix using Seaborn
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(9), yticklabels=range(9))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Print the classification report
    target_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8']
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

    print("Task B has been executed.")
