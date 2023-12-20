import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Charger le modèle
model_path = r"C:\Users\mhirs\Desktop\Nouveau dossier\modele_savedmodel"
model = tf.keras.models.load_model(model_path)

# Fonction de prétraitement de l'image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Chemin vers l'image que vous souhaitez tester
image_path = r"C:\Users\mhirs\Desktop\387518101_1581668032578290_4664452554776920562_n.jpg"

# Prétraiter l'image
preprocessed_image = preprocess_image(image_path)

# Faire la prédiction
predictions = model.predict(preprocessed_image)
predicted_class = np.argmax(predictions[0])

# Mapper l'indice de la classe à l'émotion correspondante
emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
predicted_emotion = emotion_mapping[predicted_class]

# Afficher l'image avec le résultat prédit
img = image.load_img(image_path, target_size=(48, 48), color_mode="grayscale")
plt.imshow(img, cmap='gray')
plt.title(f"Predicted Emotion: {predicted_emotion}")
plt.show()
