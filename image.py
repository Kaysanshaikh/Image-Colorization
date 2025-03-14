import logging
from flask import Flask, render_template, send_from_directory, url_for, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Flask app initialization
app = Flask(__name__)

# Configuring default values for app
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'Strange')  # Use environment variable
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Lazy loading the model
net = None
def load_model():
    """Loads the Caffe model only when needed."""
    global net
    if net is None:
        logging.info("Loading Caffe model...")
        base_dir = os.path.dirname(__file__)
        prototxt = os.path.join(base_dir, "colorization_deploy_v2.prototxt")
        caffe_model = os.path.join(base_dir, "colorization_release_v2.caffemodel")
        pts_npy = os.path.join(base_dir, "pts_in_hull.npy")

        # Ensure files exist
        if not all([os.path.exists(prototxt), os.path.exists(caffe_model), os.path.exists(pts_npy)]):
            logging.error("Model files are missing!")
            raise FileNotFoundError("One or more model files are missing. Please check paths.")
        
        # Load model
        try:
            net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
            pts = np.load(pts_npy).transpose().reshape(2, 313, 1, 1)
            net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
            net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise RuntimeError("Failed to load the model.") from e

# Form for file upload
class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(['jpg', 'jpeg', 'png'], 'Only images are allowed'),
            FileRequired('Must select an image file')
        ]
    )
    submit = SubmitField('Upload')

# Colorization function
def colorImage(image_path, image_name):
    """Processes a grayscale image and saves the colorized result."""
    try:
        load_model()  # Ensure model is loaded
        test_image = cv2.imread(image_path)
        if test_image is None:
            raise ValueError(f"Failed to read image from path: {image_path}")

        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
        normalized = test_image.astype("float32") / 255.0
        lab_image = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
        resized = cv2.resize(lab_image, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # Predict 'a' and 'b' channels
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (test_image.shape[1], test_image.shape[0]))
        L = cv2.split(lab_image)[0]
        LAB_colored = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        RGB_colored = cv2.cvtColor(LAB_colored, cv2.COLOR_LAB2RGB)
        RGB_colored = np.clip(RGB_colored, 0, 1)
        RGB_colored = (255 * RGB_colored).astype("uint8")
        RGB_BGR = cv2.cvtColor(RGB_colored, cv2.COLOR_RGB2BGR)

        # Save the result
        result_image_path = os.path.join(app.config['RESULT_FOLDER'], image_name)
        cv2.imwrite(result_image_path, RGB_BGR)
        logging.info(f"Image processed and saved at {result_image_path}")
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise RuntimeError("Image processing failed.") from e

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
    file_url, color_url = None, None
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
            logging.error(f"Error during file processing: {e}")
            return f"Error: {e}", 500
    return render_template('index.html', form=form, file_url=file_url, color_url=color_url)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=False)  # Set debug to False for production
