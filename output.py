
import os
import matplotlib.image as mpimg
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from zipfile import ZipFile
from google.colab import files
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image
from nltk.translate import meteor_score

# Install NLTK
nltk.download('punkt')
nltk.download('wordnet')

print("NLTK installation completed.")

# Load dataset and preprocess data
data_df = pd.read_csv('imgdata_with_description.csv')

print("Dataset loaded.")

# Preprocess captions
def preprocess_caption(text):
    text = str(text).lower()  # Convert to lowercase
    return text

data_df['Description'] = data_df['Description'].apply(preprocess_caption)

print("Captions preprocessed successfully.")

# Tokenize captions
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<unk>')
tokenizer.fit_on_texts(data_df['Description'])

# Ensure '<start>' token is in the tokenizer's word index
if '<start>' not in tokenizer.word_index:
    tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1

vocab_size = len(tokenizer.word_index) + 1

print("Captions tokenized successfully. Vocabulary size:", vocab_size)

# Convert text captions to sequences of integers
sequences = tokenizer.texts_to_sequences(data_df['Description'])

print("Captions converted to sequences of integers successfully.")

# Pad sequences to ensure uniform length
max_length = max(len(seq) for seq in sequences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')

print("Sequences padded successfully.")

# Prepare input data (images) and target data (captions)
image_paths = ['project/' + filename for filename in data_df['Image_File'].tolist()]
image_data = [mpimg.imread(path) for path in image_paths]
image_data = [tf.image.resize(img, (224, 224)) for img in image_data]
image_data = [img / 255.0 for img in image_data]

print("Image data prepared successfully.")

# Split data into training and validation sets
train_images, val_images, train_captions, val_captions = train_test_split(image_data, padded_sequences, test_size=0.2, random_state=42)

print("Data split into training and validation sets successfully.")

# Define CNN model for image feature extraction
def create_cnn_model():
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])
    return model

#print("CNN model created successfully.")

# Load pre-trained CNN model
cnn_model = create_cnn_model()

print("Pre-trained CNN model loaded successfully.")

# Extract features from images
train_features = cnn_model.predict(np.array(train_images))
val_features = cnn_model.predict(np.array(val_images))

print("Image features extracted successfully.")

# Define LSTM model for caption generation
caption_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(train_features.shape[1],)),  # Input layer to accept features
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.RepeatVector(max_length),  # Repeat the features to match the sequence length
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))
])

#print("LSTM model defined successfully.")

# Compile LSTM model
caption_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

#print("LSTM model compiled successfully.")

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    print("Preprocessing uploaded image...")
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.reshape((1, 224, 224, 3))  # Reshape to CNN model input shape
    print("Image preprocessed. Shape:", img.shape)
    return img

print("Image preprocessing function defined successfully.")

# Function to generate caption for the uploaded image
def generate_caption(image):
    print("Generating caption for the uploaded image...")
    img_features = cnn_model.predict(image)
    dec_input = np.zeros((1, 1))
    start_token = tokenizer.word_index.get('<start>')
    if start_token is None:
        print("Error: '<start>' token not found in the tokenizer's word index.")
        return ""
    dec_input[0, 0] = start_token
    result = []
    for i in range(max_length):
        predictions = caption_model.predict(img_features)  # Pass only image features
        predicted_id = np.argmax(predictions[0, i, :])
        if predicted_id == 0:
            break
        word = tokenizer.index_word.get(predicted_id, "<unk>")
        if word == '<end>':
            break
        result.append(word)
        dec_input[0, 0] = predicted_id
    return ' '.join(result)

print("Caption generation function defined successfully.")

# Upload image
print("Please upload an image for testing:")
uploaded_files = files.upload()

if len(uploaded_files) > 0:
    for filename in uploaded_files.keys():
        print("Processing uploaded image...")
        image_path = filename
        image = preprocess_image(image_path)

        # Get ground truth caption for testing
        ground_truth_caption = input("Enter ground truth caption: ")

        generated_caption = generate_caption(image)
        print("Generated Caption:", generated_caption)
        print("Ground Truth Caption:", ground_truth_caption)

        # Calculate METEOR score
        ground_truth_tokenized = nltk.word_tokenize(ground_truth_caption.lower())
        generated_tokenized = nltk.word_tokenize(generated_caption.lower())
        meteor_score_value = meteor_score.meteor_score([ground_truth_tokenized], generated_tokenized)
        print("METEOR Score:", meteor_score_value)

else:
    print("No image uploaded.")
