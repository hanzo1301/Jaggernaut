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
        image = tf.keras.preprocessing.image.load_img(
            os.path.join(data_dir, label, image_file),
            target_size=(224, 224)
        )
        images.append(tf.keras.preprocessing.image.img_to_array(image))
        class_labels.append(label)

images = tf.keras.applications.resnet.preprocess_input(np.array(images))
class_labels = LabelEncoder().fit_transform(class_labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, class_labels, test_size=0.2, random_state=42)

# Load pre-trained ResNet50 without top layer
base_model = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers
base_model.trainable = False

# Add classification head
inputs = tf.keras.Input(shape=(224, 224, 3))
outputs = tf.keras.layers.Dense(len(labels), activation='softmax')(tf.keras.layers.GlobalAveragePooling2D()(base_model(inputs)))
model = tf.keras.Model(inputs, outputs)

# Compile and train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.1)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", test_acc)
