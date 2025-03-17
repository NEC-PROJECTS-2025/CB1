import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Paths
model_path = "vgg16_pest_model.h5"
class_labels = ['aphids', 'armyworm', 'beetles', 'bollworm', 'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer']
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model(model_path)

# Load VGG16 for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Preprocess image function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match VGG16 input size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for VGG16
    return img_array

# Predict function
def predict_pest(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)  # Directly predict from the model
    predicted_class = np.argmax(predictions)  # Get index of highest probability
    confidence = np.max(predictions)  # Get confidence score
    return class_labels[predicted_class], confidence * 100


# Routes
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Make prediction
        predicted_class, confidence = predict_pest(file_path)

        # Pest descriptions
        descriptions = {
            'aphids': "Aphids are small sap-sucking insects that can damage plants by stunting growth and spreading diseases.",
            'armyworm': "Armyworms are destructive pests that feed on crops, causing significant agricultural damage.",
            'beetles': "Beetles are a diverse group of insects that can be harmful to plants, wood, and stored products.",
            'bollworm': "Bollworms are larvae of moths that attack cotton and other crops, causing economic losses.",
            'grasshopper': "Grasshoppers are herbivorous insects that can devastate crops by consuming large amounts of foliage.",
            'mites': "Mites are tiny arachnids that can damage plants by feeding on their sap and causing leaf discoloration.",
            'mosquito': "Mosquitoes are flying insects known for spreading diseases like malaria and dengue through their bites.",
            'sawfly': "Sawflies are insects whose larvae feed on plants, often causing defoliation and reduced crop yields.",
            'stem_borer': "Stem borers are larvae of moths that tunnel into plant stems, weakening and killing the plants."
        }

        # Get the pest description based on the predicted class
        pest_description = descriptions.get(predicted_class, "No description available.")

        # Render the result
        return render_template("index.html", 
                               predicted_class=predicted_class, 
                               confidence=f"{confidence:.2f}", 
                               image_filename=filename,
                               pest_description=pest_description)

if __name__ == "__main__":
    app.run(debug=True)