from flask import Flask, render_template, send_from_directory, url_for, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
import psutil  # For monitoring memory usage
import gdown

# Download the model file from Google Drive
url = "https://drive.google.com/uc?id=1LAH7CvTjwswtT6OLLKTiojCoARP1o_mL"
output = "colorization_release_v2.caffemodel"
gdown.download(url, output, quiet=False)

# Flask app initialization
app = Flask(__name__)

# Configurations
app.config['SECRET_KEY'] = 'Strange'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Upload form for image submission
class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(['jpg', 'jpeg', 'png'], 'Only images are allowed'),
            FileRequired('Must select an image file')
        ]
    )
    submit = SubmitField('Upload')

# Utility to log memory usage
def log_memory_usage(context=""):
    memory = psutil.virtual_memory()
    print(f"{context} - Memory Usage: {memory.percent}% used, {memory.available // (1024**2)} MB available")

# Colorization function
def colorImage(image_path, image_name):
    log_memory_usage("Before loading model files")
    
    # Paths to model files
    base_dir = os.path.dirname(__file__)
    prototxt = os.path.join(base_dir, "colorization_deploy_v2.prototxt")
    caffe_model = os.path.join(base_dir, "colorization_release_v2.caffemodel")
    pts_npy = os.path.join(base_dir, "pts_in_hull.npy")

    # Ensure all files exist
    if not (os.path.exists(prototxt) and os.path.exists(caffe_model) and os.path.exists(pts_npy)):
        raise FileNotFoundError("Missing one or more required files: Prototxt, CaffeModel, or pts_in_hull.npy")

    # Load the model and cluster centers
    net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
    pts = np.load(pts_npy)
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    log_memory_usage("After loading model files")

    # Read and process the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to read the input image.")
    
    # Resize the image to reduce memory usage
    image = cv2.resize(image, (512, 512))  # Resized to 512x512 for optimization

    h, w = image.shape[:2]
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = image_lab[:, :, 0]
    l_channel_resized = cv2.resize(l_channel, (224, 224))
    l_channel_resized -= 50

    net.setInput(cv2.dnn.blobFromImage(l_channel_resized))
    ab_channels = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channels = cv2.resize(ab_channels, (w, h))

    lab_colored = np.concatenate((l_channel[:, :, np.newaxis], ab_channels), axis=2)
    bgr_colored = cv2.cvtColor(lab_colored, cv2.COLOR_LAB2BGR)
    bgr_colored = np.clip(bgr_colored, 0, 255).astype("uint8")

    result_path = os.path.join(app.config['RESULT_FOLDER'], image_name)
    cv2.imwrite(result_path, bgr_colored)

    log_memory_usage("After processing image")

# Routes
@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def get_color_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    file_url = None
    color_url = None

    if form.validate_on_submit():
        file = form.photo.data
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            colorImage(file_path, filename)
            file_url = url_for('get_file', filename=filename)
            color_url = url_for('get_color_file', filename=filename)
        except Exception as e:
            return f"Error: {e}", 500

    return render_template('index.html', form=form, file_url=file_url, color_url=color_url)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/results')
def resultImage():
    original_images = '/uploads/'
    colored_images = '/results/'
    return render_template('result.html', file=original_images, color_file=colored_images)

if __name__ == "__main__":
    app.run(debug=True, threaded=False)  # Reduced threads for optimized resource usage
