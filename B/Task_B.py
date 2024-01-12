
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
    # Initialize a consistent random state for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Replace with the path to your PathMNIST dataset
    file_path = 'Datasets/pathmnist.npz'

    # Load the dataset from specified path
    data = np.load(file_path)

    # Split the dataset into training, validation, and test sets
    train_images, val_images, test_images = data['train_images'], data['val_images'], data['test_images']
    train_labels, val_labels, test_labels = data['train_labels'], data['val_labels'], data['test_labels']

    # Normalize image data to the [0, 1] range
    train_images, val_images, test_images = train_images / 255.0, val_images / 255.0, test_images / 255.0

    # Convert integer labels into one-hot encoded vectors
    num_classes = 9
    train_labels = to_categorical(train_labels, num_classes)
    val_labels = to_categorical(val_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    # Define the CNN architecture with VGG-like layers
    model = Sequential([
        # Convolutional layers to extract features; using ReLU activation and same padding
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 3)),
        # Doubling the number of filters with each convolutional block
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        # Max pooling to reduce spatial dimensions of the feature maps
        MaxPooling2D(2, 2),
        # Dropout layers to mitigate overfitting by randomly dropping out nodes during training
        Dropout(0.3),
        Flatten(),
        # Dense layer for classification
        Dense(128, activation='relu'),
        Dropout(0.5),
        # Output layer with softmax activation for multi-class probability distribution
        Dense(num_classes, activation='softmax')
    ])

    # Implement early stopping to halt training when validation loss doesn't improve
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # Compile the model with the Adam optimizer and categorical crossentropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model on the training data, validating on validation data
    history = model.fit(
        train_images, train_labels,
        batch_size=64, epochs=30,
        validation_data=(val_images, val_labels),
        callbacks=[early_stopping]
    )

    # Plot training/validation loss and accuracy to assess model performance over epochs
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

    # Evaluate the model's performance on the test dataset
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test loss: {test_loss}\nTest accuracy: {test_acc}")

    # Predict on test dataset and generate a confusion matrix and classification report
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)

    # Visualize the confusion matrix using Seaborn's heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Output a detailed classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=[f'Class {i}' for i in range(num_classes)]))

    print("Task B has been executed.")
