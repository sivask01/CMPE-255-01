# Project: Mask Guard
#### CMPE-255-01
#### Group 12
#### San Jose State University, Fall 2023

# Objective
The objective of the MaskGuardPro project is to develop a sophisticated image processing and machine learning system that can accurately detect whether individuals in public spaces, such as hospitals, restaurants, and malls, are wearing face masks and, crucially, whether these masks are worn correctly. This technology aims to enhance public health measures by ensuring compliance with mask-wearing protocols during the COVID-19 pandemic and beyond.

# Model

The finalized architecture for MaskGuardPro is a Convolutional Neural Network (CNN) as depicted in Figure 2.1. This CNN architecture is designed with multiple convolutional layers, each followed by max pooling layers, to effectively extract and downsample spatial hierarchies of features from the input images. The initial convolutional layers capture basic patterns such as edges and textures, while deeper layers detect more complex features relevant to mask detection.
The network employs strides to reduce dimensionality and max pooling for spatial variance reduction, which enhances the model's ability to handle variations in mask positioning and fit. Towards the end of the network, densely connected layers consolidate the high-level features extracted by the convolutional layers, culminating in a final output layer that classifies the images into the respective categories of mask wearing.
This architecture is meticulously optimized to balance the computational load while maintaining high accuracy in identifying correct mask usage, ensuring the system’s viability for real-time applications in diverse settings.


![alt text](assets/CNN%20model.png)

# Requirements
    ! pip install requirements.txt

# Usage
### Optional

1. Importing the Required Module:
from google.colab import drive: This line imports the drive module from the google.colab library. Google Colab provides this library as part of its environment, and it includes various functions to integrate Colab notebooks with other Google services like Drive.

2. Mounting Google Drive:
drive.mount('/content/drive'): This line mounts the user's Google Drive to the Colab runtime environment. When this code is executed, it prompts the user to authorize access to their Google Drive. This is done through a link that leads to a sign-in page, where the user grants permission for the Colab notebook to access their Drive.

    from google.colab import drive
    drive.mount('/content/drive')

## Data Preparation

Masks play a crucial role in protecting the health of individuals against respiratory diseases, as is one of the few precautions available for COVID-19 in the absence of immunization. With this dataset, it is possible to create a model to detect people wearing masks, not wearing them, or wearing masks improperly.
This dataset contains 853 images belonging to the 3 classes, as well as their bounding boxes in the PASCAL VOC format.

The classes are:
With mask;
Without mask;
Mask worn incorrectly.

download data from: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

Make sure the data is in this format

    data/
    ├── classes/
    ├── CMFD/ # Correctly Masked Face Dataset
    ├── IMFD/ # Incorrectly Masked Face Dataset
    └── NO_MASK/ # No Mask


## Data Loading
The data loading process involves importing necessary libraries and defining a function to display images. This function, display_images, is used to show a specified number of images from each category in the dataset. The dataset appears to have three categories: Correctly Masked Face Dataset (CMFD), Incorrectly Masked Face Dataset (IMFD), and No Mask. Here's an example of how to use the function to display images from each category:

Data can be download from 

    import os
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # Function to display images
    def display_images(folder_path, num_images=3):
        image_files = os.listdir(folder_path)[:num_images]
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

        for ax, image_file in zip(axes, image_files):
            image_path = os.path.join(folder_path, image_file)
            img = mpimg.imread(image_path)
            ax.imshow(img)
            ax.axis('off')
        plt.show()

    # Displaying images from each category
    print("Images from CMFD (Correctly Masked Face Dataset):")
    display_images(f"{extracted_folder_path}/Classes/CMFD")

    print("Images from IMFD (Incorrectly Masked Face Dataset):")
    display_images(f"{extracted_folder_path}/Classes/IMFD")

    print("Images from No Mask:")
    display_images(f"{extracted_folder_path}/Classes/NO_MASK")


### Data Pre Processing

In our face mask classifier project, we perform several key preprocessing steps to prepare our image data for training a neural network. These steps are implemented using TensorFlow's Keras API and include image resizing, normalization, and data augmentation. Here's an overview of the process:

1. Setting Up Parameters:
We define a standard image_size of (64, 64) pixels for all images. This ensures uniformity in the input size for the model.
The batch_size is set to 16, meaning the model will process 16 images at a time during training.

2. ImageDataGenerator:
We use the ImageDataGenerator class for easy and efficient data augmentation and preprocessing.
The generator is set to rescale pixel values from [0, 255] to [0, 1] (rescale=1./255), a common practice for neural network inputs as it helps in faster convergence.
We also specify a validation_split of 0.2, meaning 20% of the data will be reserved for validation purposes.

3. Training and Validation Generators:
The flow_from_directory method creates two generators: one for training and one for validation.
These generators automatically label images based on the directory structure (assumed to be extracted_folder_path/Classes), making it convenient to work with labeled datasets.
The target_size is set according to the predefined image_size.
class_mode is set to 'categorical', suitable for multi-class classification tasks.
The training generator uses the subset='training' portion of the data, while the validation generator uses subset='validation'.

4. Class Indices:
Finally, we retrieve class_indices from the training generator, which gives us a mapping of class names to their numeric labels. This is useful for understanding model predictions and evaluating performance.
These preprocessing steps are essential for setting up a robust pipeline that feeds normalized and augmented image data into the model, enhancing its ability to learn and generalize from the training data.

    
### Creating a Model (CNN )

Our project uses a Convolutional Neural Network (CNN) model, constructed using TensorFlow's Keras Sequential API. The model is specifically designed for image classification and includes the following layers:

1. Convolutional Layers:
The model begins with a Conv2D layer of 32 filters of size (3, 3) and 'relu' activation. This layer is responsible for extracting features from the input images. It is followed by another Conv2D layer with 64 filters, also with 'relu' activation. Increasing the number of filters helps the network learn more complex features.

2. MaxPooling Layers:
Each convolutional layer is followed by a MaxPooling2D layer with a pool size of (2, 2). These layers reduce the spatial dimensions (width and height) of the output volume, helping to reduce the computation and the number of parameters.

3. Flattening:
The Flatten layer is used to convert the 2D feature maps into a 1D feature vector. This step is necessary before feeding the data into the dense layers for classification.

4. Dense Layers:
A Dense layer with 64 units and 'relu' activation is used. This layer is the fully connected layer that uses the features learned during the convolutional steps for classification.
The Dropout layer with a dropout rate of 0.5 is included to prevent overfitting. It randomly sets a fraction of input units to 0 at each update during training time, which helps to prevent overfitting.

5. Output Layer:
The final layer is a Dense layer with 3 units and a 'softmax' activation function. The 3 units correspond to the three classes we aim to classify, and the 'softmax' activation enables the model to output a probability distribution over the three classes.

This architecture is well-suited for image classification tasks where the input is a set of images, and the goal is to categorize them into distinct classes. In our case, the model is designed to distinguish between three different classes, likely representing different types of face mask usage.



    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        # Dense(1, activation='sigmoid')  # Binary classification
        Dense(3, activation='softmax')  # For three classes
    ])

### Compile and Train the Model

### Compilation

Before training, the model is compiled with specific settings for optimization and loss calculation:

1. Optimizer: We use the 'adam' optimizer. Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments. It's well-suited for large datasets and high-dimensional spaces.

2. Loss Function: For the loss function, we use 'categorical_crossentropy' since our task is a multi-class classification problem. This loss function is appropriate when there are two or more label classes. We use it to compare the predicted label with the true label of the image.

3. Metrics: The metric used to evaluate the performance of the model is 'accuracy', which is a common choice for classification problems.

### Training

The model is trained using the fit method, which takes the following parameters:

train_generator: The training data generator, which feeds the model with training data in batches, as specified by the batch_size parameter during its creation.
steps_per_epoch: This is set to the total number of samples in our training set divided by the batch size. It determines the number of batch steps before declaring one epoch finished and starting the next epoch.
validation_data: The validation data generator is used to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data.
validation_steps: Similar to steps_per_epoch, but for the validation data.
epochs: The number of epochs to train the model. An epoch is an iteration over the entire training data. In this case, it's set to 2, but this can be adjusted based on the model's performance and the available computational resources.
The history object returned by fit keeps a record of the loss values and metric values during training. This data is useful for analyzing the training process, especially for understanding overfitting and underfitting.

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=2 # Adjust as necessary
    )

# Model Testing

This code snippet demonstrates the process of loading, preprocessing, and predicting the class of images from a test dataset using a trained model in TensorFlow.

1. Test Image Loading: It starts by defining a path to the test images and then creates a list of all .jpg image paths within that directory.
2. Prediction Function: A function predict_image is defined to handle the loading and preprocessing of each image. This includes resizing the image to 64x64 pixels (the input size expected by the model), converting it to an array, normalizing pixel values to the range [0, 1], and adding a batch dimension.
3. Model Prediction: The function then uses the trained model to predict the class of the image. It returns the index of the highest probability, which corresponds to the predicted class.
4. Mapping Predictions to Class Labels: The script maps these indices to actual class labels ('CMFD', 'IMFD', 'No Mask').
5. Iterating Over Test Images: Finally, the script iterates over all images in the test dataset, uses the predict_image function to get the predicted class for each image, and prints out the file name along with its predicted class.

This process is essential for evaluating the model's performance on unseen data and provides a practical way to deploy the model for real-world testing.

    import os
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import numpy as np
    test_folder_path = f"{extracted_folder_path}/Test_Classes"
    test_images = [os.path.join(test_folder_path, file) for file in os.listdir(test_folder_path) if file.endswith('.jpg')]
    def predict_image(image_path, model):
        # Load and preprocess the image
        img = load_img(image_path, target_size=(64, 64))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(img_array)
        # print("index of highest", np.argmax(prediction, axis=1))
        return np.argmax(prediction, axis=1)  # Returns the index of the highest probability
    class_labels = ['CMFD', 'IMFD', 'No Mask']
    for image_path in test_images:
        predicted_index = predict_image(image_path, model)[0]
        class_prediction = class_labels[predicted_index]
        print(f"Image: {os.path.basename(image_path)}, Prediction: {class_prediction}")

# Citation
    
    This project is based on the work of:
    {
        "entity": "San Jose State University",
        "title": "Mask Guard Pro",
        "session": "Fall 2023",
        "Class": "CMPE 255-01 Data Mining",
        "Author": [
        Yongen Chen,
        Bunpheng Chhay,  
        Kevin Ho,          
        Sivakrishna Yaganti 
        ]

    }