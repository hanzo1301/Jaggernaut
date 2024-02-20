import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data_dir = "path_to_dataset_directory"
labels = os.listdir(data_dir)
images = []
class_labels = []

for label in labels:
    for image_file in os.listdir(os.path.join(data_dir, label)):
        images.append(os.path.join(data_dir, label, image_file))
        class_labels.append(label)

class_labels = LabelEncoder().fit_transform(class_labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, class_labels, test_size=0.2, random_state=42)

# Load pre-trained Facenet model
facenet_model = tf.keras.models.load_model("path_to_pretrained_facenet_model")

# Preprocess and embed images using Facenet
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.facenet.preprocess_input(img)
    return img

X_train_embeddings = np.array([facenet_model.predict(preprocess_image(img_path))[0] for img_path in X_train])
X_test_embeddings = np.array([facenet_model.predict(preprocess_image(img_path))[0] for img_path in X_test])

# Build and train classifier
classifier = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X_train_embeddings, y_train, epochs=10, validation_split=0.1)

# Evaluate classifier
test_loss, test_acc = classifier.evaluate(X_test_embeddings, y_test)
print("\nTest Accuracy:", test_acc)
