import cv2
import numpy as np
import tensorflow as tf

# Load MobileNet model and labels
model = tf.keras.applications.MobileNetV2(weights='imagenet')
labels_path = tf.keras.utils.get_file(
    'imagenet_class_index.json',
    'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
)
with open(labels_path) as f:
    labels = {int(key): value for (key, value) in json.load(f).items()}

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)

def preprocess_image(image):
    image_resized = cv2.resize(image, (224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

def get_label(pred):
    idx = np.argmax(pred)
    return labels[idx][1]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame for MobileNet
    processed_frame = preprocess_image(frame)
    
    # Make prediction
    predictions = model.predict(processed_frame)
    
    # Get the label of the highest confidence prediction
    label = get_label(predictions[0])
    
    # Display the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('MobileNet Object Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
