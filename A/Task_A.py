import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

def run_task_a():
    # Load the dataset
    # The training, validation, and test sets of the NumPy file containing the PneumoniaMNIST dataset are loaded.
    # These datasets are used, in turn, to train the model, adjust its parameters, and assess its effectiveness.

    data = np.load(r'Datasets\pneumoniamnist.npz')
    train_images = data['train_images']  
    val_images = data['val_images']      
    test_images = data['test_images']    
    train_labels = data['train_labels']  
    val_labels = data['val_labels']      
    test_labels = data['test_labels']    

    # Data preprocessing
    # Pre-processing of image data including reshaping and normalisation.
    # The reshaping operation ensures that the image has the correct dimensions, and normalisation helps the model learn faster and better.

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    val_images = val_images.reshape(val_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    train_images = train_images.astype('float32') / 255
    val_images = val_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # Visualisation of training images
    # This function is used to display images from the training dataset
    # It helps us visualise what the data looks like and labels.

    def plot_images(images, labels, rows=4, cols=5):
        plt.figure(figsize=(10, 8))
        for i in range(rows * cols):
            plt.subplot(rows, cols, i+1)
            plt.imshow(images[i].reshape(28, 28), cmap='gray')
            plt.title('Pneumonia' if labels[i] else 'Normal')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    plot_images(train_images, train_labels)

    # Data Enhancement Configuration
    # In order to improve the generalisation of the model, we have used data augmentation techniques here.
    # These techniques can increase the diversity of the data by slightly altering the way the training images are presented (e.g., rotating, scaling, etc.).

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )

    # Building Convolutional Neural Network(CNN) Models
    # We use the Sequential model, which is a simple linear stacking model that is perfect for beginners.
    # The Sequential model allows us to simply add network layers one at a time.

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # First convolutional layer, 32 filters, each of size 3x3.
        MaxPooling2D(2, 2),  # First pooling layer with pooling window size 2x2.
        Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer, 64 filters.
        MaxPooling2D(2, 2),  # Second pooling layer.
        Conv2D(128, (3, 3), activation='relu'), # Third convolutional layer, 128 filters.
        MaxPooling2D(2, 2),  # Third pooling layer.
        Flatten(),  # Spreading layer for converting the 2D feature maps output from the previous convolutional layer to 1D for use in the fully connected layer.
        Dropout(0.25),  # Dropout layer that randomly drops some neurons at a rate of 0.25 to reduce overfitting.
        Dense(128, activation='relu'),  # Fully connected layer with 128 neurons.
        Dense(1, activation='sigmoid')  # Output layer, 1 neuron, using sigmoid activation function for binary classification problems.
    ])

    # Early stopping is used to prevent model overfitting. If the validation loss does not improve in 15 consecutive epochs, training is stopped.

    early_stopping = EarlyStopping(monitor='val_loss', patience=15)

    # compilation model# We use the Adam optimiser, binary cross-entropy loss function and accuracy as performance evaluation metrics.

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    # Here we use the fit method to train the model. We use the previously defined data augmentation, as well as validation data and early stop callbacks.

    history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                        epochs=100,
                        validation_data=(val_images, val_labels),
                        callbacks=[early_stopping])

    # Plot loss and accuracy curves during training
    # These graphs can help us understand how the model is performing during training, including whether it is overfitting or if more training is needed.

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    # Test and evaluate the model
    # After the model has been trained, we evaluate the performance of the model on a test set.

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")

    # Generate confusion matrices and classification reports
    # We use the confusion matrix to visualise how the model performs on different categories.

    threshold = 0.6
    predictions = model.predict(test_images)
    predictions = [1 if x > threshold else 0 for x in predictions]

    # Use Seaborn to draw confusion matrices

    cm = confusion_matrix(test_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Print the classification report
    # The Classification Report provides detailed metrics such as Precision, Recall and F1 Score for each category.

    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=['Normal', 'Pneumonia']))
