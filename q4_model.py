# -*- coding: utf-8 -*-

import requests
import os
from bs4 import BeautifulSoup
import urllib

# Create the folder if it doesn't exist
folder_name = "Garfield"
os.makedirs(folder_name, exist_ok=True)

# URL of the website to scrape
url = "http://pt.jikos.cz/garfield/2023/"

# Send a GET request to the website
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find all image tags on the page
image_tags = soup.find_all("img")

# Download and save each image
for image_tag in image_tags:
    try:
        image_url = image_tag["src"]
        image_name = image_url.split("/")[-1]
        image_path = os.path.join(folder_name, image_name)
        urllib.request.urlretrieve(image_url, image_path)
    except ValueError:
        print("Error retrieving image URL:", image_url)

#converting the gif images to jpg

import os
from PIL import Image

# Specify the folder path
folder_path = '/Users/cefiljosephsoans/Desktop/Projects and Applications of Data Science/Programming Final/Question 4/Dataset'

# Get the list of files in the folder
files = os.listdir(folder_path)

# Iterate over each file
for file in files:
    # Get the file extension
    file_extension = os.path.splitext(file)[1]

    # Check if the file is a GIF
    if file_extension.lower() == '.gif':
        # Get the current file path
        current_path = os.path.join(folder_path, file)

        # Open the GIF image
        gif_image = Image.open(current_path)

        # Convert the GIF image to JPG format
        jpg_image = gif_image.convert('RGB')

        # Create the new file name with the JPG extension
        new_file_name = os.path.splitext(file)[0] + '.jpg'

        # Get the new file path
        new_path = os.path.join(folder_path, new_file_name)

        # Save the JPG image
        jpg_image.save(new_path)

        # Close the GIF and JPG images
        gif_image.close()
        jpg_image.close()


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Resizing


batch_size = 32
img_height = 224
img_width = 224

train_dataset = image_dataset_from_directory(
    'Dataset',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validation_dataset = image_dataset_from_directory(
    'Dataset',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_dataset.class_names
print(class_names)

#given parameters
batch_size = 1000
img_height = 224
img_width = 224

dataset_path = 'Dataset'
dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

#loop to get image features from the dataset
images, labels = [], []
for image_batch, label_batch in dataset.as_numpy_iterator():
    images.append(image_batch)
    labels.append(label_batch)

images = np.concatenate(images)
labels = np.concatenate(labels)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

#augmentation function
def augment_data(x, y):
    x = tf.image.resize(x, (img_height, img_width))
    x = tf.image.rot90(x, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return x, y

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.map(augment_data)

train_dataset = train_dataset.batch(batch_size).map(lambda x, y: (x / 255.0, y))

#loop to print the images
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(min(16, batch_size)):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype('float32'))
        plt.title(class_names[labels[i]])
        plt.axis('off')

plt.suptitle("Example Real-World Images from the Dataset", fontsize=16)
plt.show()


local_weights_file = "/Users/cefiljosephsoans/Desktop/Projects and Applications of Data Science/Programming Final/Question 4/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
model_resize = Sequential([Resizing(224,224),ResNet50(include_top= False, weights = local_weights_file,pooling = "max", classes = 2)])


train_images = []
train_labels = []

for img, lab in train_dataset:
    train_images.append(img)
    train_labels.append(lab)

train_images = np.concatenate(train_images, axis=0)
train_labels = np.concatenate(train_labels, axis=0)

train_features = model_resize.predict(train_images)

svc = SVC()
svc.fit(train_features, train_labels)

validation_images = []
validation_labels = []

for img, lab in validation_dataset:
    validation_images.append(img)
    validation_labels.append(lab)

validation_images = np.concatenate(validation_images, axis=0)
validation_labels = np.concatenate(validation_labels, axis=0)

validation_features = model_resize.predict(validation_images)

svc = SVC()
svc.fit(validation_features, validation_labels)

y_pred = svc.predict(validation_features)

cm = confusion_matrix(y_pred,validation_labels)

disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(class_names))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Validation Set")
plt.show()

report = classification_report(y_pred,validation_labels)

accuracy = accuracy_score(y_pred,validation_labels)

print(f"Accuracy: {accuracy}")

report = classification_report(y_pred,validation_labels)
print("Classification Report:")
print(report)

incorrect_images = []
for i in range(len(y_pred)):
    if(y_pred[i]!= validation_labels[i]):
        plt.imshow(validation_images[i].astype(np.uint8))
        plt.title(f"Predicted:{class_names[y_pred[i]]}\n Actual:{class_names[validation_labels[i]]}")
        plt.show()

incorrect_images = []
for i in range(5):
    if(y_pred[i] == validation_labels[i]):
        plt.imshow(validation_images[i].astype(np.uint8))
        plt.title(f"Predicted:{class_names[y_pred[i]]}\n Actual:{class_names[validation_labels[i]]}")
        plt.show()

#no wrongly classified ones

# Import the necessary libraries
import streamlit as st
import numpy as np
from PIL import Image
import streamlit as st
import numpy as np
from PIL import Image
from sklearn.svm import SVC

# Load the trained model
model = SVC()

# Fit the model with appropriate arguments
model.fit(train_features, train_labels)

# Define the Streamlit app
def main():
    # Set the title and description of the app
    st.title("Manga or Comic?")
    st.write("Upload an image and the model will make a prediction.")

    # Create a file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Make a prediction when an image is uploaded
    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)

        # Preprocess the image
        image = image.resize((224, 224))  # Resize the image to match the input size of the model
        image = np.array(image)  # Convert the image to a numpy array
        image = image.reshape(1, -1)  # Flatten the image to a 2-dimensional array

        # Make a prediction using the trained model
        prediction = model.predict(image)

        # Display the prediction
        st.image(image, caption=f"Prediction: {prediction}")

# Run the Streamlit app
if __name__ == "__main__":
    main()