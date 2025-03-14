from flask import Flask, render_template, send_from_directory, url_for, request
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os

# Flask app configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Paths to model files
prototxt = 'model/colorization_deploy_v2.prototxt'
model = 'model/colorization_release_v2.caffemodel'
points = 'model/pts_in_hull.npy'

# Ensure all model files are available
if not os.path.isfile(model):
    raise FileNotFoundError("Model file is missing. Download it from the README link and place it in the 'model/' folder.")

# Load pretrained model
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points).transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the image.")

    h, w = image.shape[:2]
    image = cv2.resize(image, (512, 512))  # Resize to reduce memory usage
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = cv2.split(lab)[0]
    l_channel_resized = cv2.resize(l_channel, (224, 224))
    l_channel_resized -= 50

    net.setInput(cv2.dnn.blobFromImage(l_channel_resized))
    ab_channels = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channels = cv2.resize(ab_channels, (w, h))

    l_channel = cv2.split(lab)[0]
    lab_colored = np.concatenate((l_channel[:, :, np.newaxis], ab_channels), axis=2)
    bgr_colored = cv2.cvtColor(lab_colored, cv2.COLOR_LAB2BGR)
    bgr_colored = np.clip(bgr_colored, 0, 255).astype("uint8")
    cv2.imwrite(output_path, bgr_colored)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    file.save(file_path)

    try:
        colorize_image(file_path, result_path)
        return send_from_directory(app.config['RESULT_FOLDER'], filename)
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
