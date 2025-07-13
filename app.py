# Importing all necessary modules and libraries
from flask import Flask, render_template, send_from_directory, url_for, request, redirect
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initiating app with name 'app'
app = Flask(__name__)

# Configuring default values for app
app.config['SECRET_KEY'] = 'Strange'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF protection
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Class for form upload validation and uploads
class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(['jpg', 'jpeg', 'png'], 'Only images are allowed'),
            FileRequired('Must select an image file')
        ]
    )
    submit = SubmitField('Upload')

# Function to check if model files exist
def check_model_files():
    base_dir = os.path.dirname(__file__)
    prototxt = os.path.join(base_dir, "colorization_deploy_v2.prototxt")
    caffe_model = os.path.join(base_dir, "colorization_release_v2.caffemodel")
    pts_npy = os.path.join(base_dir, "pts_in_hull.npy")
    
    missing_files = []
    if not os.path.exists(prototxt):
        missing_files.append("colorization_deploy_v2.prototxt")
    if not os.path.exists(caffe_model):
        missing_files.append("colorization_release_v2.caffemodel")
    if not os.path.exists(pts_npy):
        missing_files.append("pts_in_hull.npy")
    
    return missing_files

# Function to convert greyscale image to RGB and save it into result directory
def colorImage(image_path, image_name):
    # Define paths for model files
    base_dir = os.path.dirname(__file__)
    prototxt = os.path.join(base_dir, "colorization_deploy_v2.prototxt")
    caffe_model = os.path.join(base_dir, "colorization_release_v2.caffemodel")
    pts_npy = os.path.join(base_dir, "pts_in_hull.npy")

    # Check if files exist
    missing_files = check_model_files()
    if missing_files:
        raise FileNotFoundError(
            f"Missing model files: {', '.join(missing_files)}\n"
            f"Download them from: https://github.com/richzhang/colorization/tree/caffe"
        )

    try:
        # Load the model
        net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
        pts = np.load(pts_npy)
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    except Exception as e:
        raise RuntimeError(f"Error loading model files: {e}")

    try:
        # Processing the image
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

        # Predicting a and b values
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (test_image.shape[1], test_image.shape[0]))

        # Combining L, a, and b channels
        L = cv2.split(lab_image)[0]
        LAB_colored = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        RGB_colored = cv2.cvtColor(LAB_colored, cv2.COLOR_LAB2RGB)
        RGB_colored = np.clip(RGB_colored, 0, 1)
        RGB_colored = (255 * RGB_colored).astype("uint8")
        RGB_BGR = cv2.cvtColor(RGB_colored, cv2.COLOR_RGB2BGR)

        # Saving the image
        result_image_path = os.path.join(app.config['RESULT_FOLDER'], image_name)
        cv2.imwrite(result_image_path, RGB_BGR)
        app.logger.info(f"Successfully colorized image: {image_name}")
    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")

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
    
    # Check if model files exist
    missing_files = check_model_files()
    if missing_files:
        error_msg = f"Missing model files: {', '.join(missing_files)}. Please download them from the colorization repository."
        return render_template('index.html', form=form, error=error_msg)
    
    if request.method == 'POST':
        app.logger.debug(f"POST request received. Form data: {request.form}")
        app.logger.debug(f"Files: {request.files}")
        
        if form.validate_on_submit():
            file = form.photo.data
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(file_path)
                app.logger.info(f"File saved: {filename}")
                
                colorImage(file_path, filename)
                app.logger.info(f"Image colorized successfully: {filename}")
                
                file_url = url_for('get_file', filename=filename)
                color_url = url_for('get_color_file', filename=filename)
                
                return render_template('index.html', form=form, file_url=file_url, color_url=color_url)
            except Exception as e:
                app.logger.error(f"Error processing image: {e}")
                return render_template('index.html', form=form, error=str(e))
        else:
            app.logger.error(f"Form validation failed. Errors: {form.errors}")
            return render_template('index.html', form=form, error="Form validation failed. Please check your file.")
    
    return render_template('index.html', form=form)

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
    app.run(host="0.0.0.0", port=7860, debug=True)
