import os
import matplotlib.image as mpimg
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from zipfile import ZipFile
from google.colab import files
import pickle
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image

# Install required packages
!pip install kaggle

# Upload Kaggle API token
files.upload()


# Move API token to correct directory
!mkdir -p /root/.kaggle
!mv kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

# Set Kaggle config directory
os.environ['KAGGLE_CONFIG_DIR'] = '/root/.kaggle'

# Download megha1 image dataset
!kaggle datasets download -d meghabprathap/project

# Unzip dataset
with ZipFile('project.zip', 'r') as zip_ref:
    zip_ref.extractall('project')

# Create CSV file
image_files = [f for f in os.listdir('project') if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# Write image data to CSV file
with open('imgdata_with_description.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image_File', 'Width', 'Height', 'Description'])
    for image_file in image_files:
        img = mpimg.imread(os.path.join('project', image_file))
        height, width, _ = img.shape
        image_name = os.path.basename(image_file)

        # Check if a corresponding text file exists
        text_file_path = os.path.join('project', os.path.splitext(image_name)[0] + '.txt')
        if os.path.exists(text_file_path):
            # Read the content of the text file as description
            with open(text_file_path, 'r') as text_file:
                description = text_file.read()
        else:
            description = ""

        writer.writerow([image_name, width, height, description])

# Load CSV file containing image data and descriptions
data_df = pd.read_csv('imgdata_with_description.csv')

# Preprocessing function for captions
def preprocess_caption(text):
    text = str(text)  # Convert to string
    text = text.strip()  # Remove leading and trailing whitespaces
    return text

# Apply preprocessing to captions
data_df['Description'] = data_df['Description'].apply(preprocess_caption)

# Tokenize captions
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<unk>')
tokenizer.fit_on_texts(data_df['Description'])
vocab_size = len(tokenizer.word_index) + 1

# Convert text captions to sequences of integers
sequences = tokenizer.texts_to_sequences(data_df['Description'])

# Pad sequences to ensure uniform length
max_length = max(len(seq) for seq in sequences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')

# Check if preprocessing is successful
if len(sequences) > 0 and len(padded_sequences) > 0:
    print("Preprocessing successful.")
else:
    print("Error: Preprocessing failed.")

# Prepare input data (images) and target data (captions)
image_paths = ['project/' + filename for filename in data_df['Image_File'].tolist()]
image_data = [tf.io.read_file(path) for path in image_paths]
image_data = [tf.image.decode_jpeg(img, channels=3) for img in image_data]
image_data = [tf.image.resize(img, (224, 224)) for img in image_data]
image_data = [img / 255.0 for img in image_data]

# Split data into training and validation sets
train_images, val_images, train_captions, val_captions = train_test_split(image_data, padded_sequences, test_size=0.2, random_state=42)

# Save Tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Define CNN model for image feature extraction
def create_cnn_model():
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])
    return model

# Load pre-trained CNN model
cnn_model = create_cnn_model()

# Compile CNN model
print("Compiling CNN model...")
cnn_model.compile(optimizer='adam', loss='mse')  # You need to define your appropriate optimizer and loss function
print("CNN model compilation successful.")

# Extract features from images
train_features = cnn_model.predict(np.array(train_images))
val_features = cnn_model.predict(np.array(val_images))

# Define LSTM model for caption generation
print("Building LSTM model...")
caption_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(train_features.shape[1],)),  # Input layer to accept features
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.RepeatVector(max_length),  # Repeat the features to match the sequence length
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))
])
print("LSTM model built successfully.")

# Compile LSTM model
print("Compiling LSTM model...")
caption_model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))
print("LSTM model compilation successful.")

# Check if LSTM model construction is successful
caption_model.summary()

# Train the CNN model
print("Training CNN model...")
cnn_history = cnn_model.fit(x=np.array(train_images), y=np.array(train_features), epochs=10, validation_data=(np.array(val_images), np.array(val_features)), verbose=1)
print("Training CNN model completed successfully.")

# Train the LSTM model
print("Training LSTM model...")
lstm_history = caption_model.fit(x=train_features, y=train_captions, epochs=10, validation_data=(val_features, val_captions), verbose=1)
print("Training LSTM model completed successfully.")

# Save CNN model
cnn_model.save('saved_models/cnn_model.h5')

# Save LSTM model
caption_model.save('saved_models/lstm_model.h5')

print("CNN and LSTM models saved successfully.")
