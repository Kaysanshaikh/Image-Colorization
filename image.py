# Importing all necessary modules and libraries
from flask import Flask, render_template, send_from_directory, url_for, request, redirect
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import numpy as np
import cv2
import os
import gc
import psutil
import threading
from werkzeug.utils import secure_filename

# Initiating app with name 'app'
app = Flask(__name__)

# Configuring default values for app
app.config['SECRET_KEY'] = 'Strange'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB max upload size
app.config['MAX_IMAGE_SIZE'] = 512  # pixels - reduced from 1024

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Global variables for model
net = None
pts = None
model_lock = threading.Lock()

# Function to load the model - now using lazy loading
def get_model():
    global net, pts
    
    # Only load the model if it's not already loaded
    if net is None:
        with model_lock:  # Thread safety
            if net is None:  # Double-check
                try:
                    base_dir = os.path.dirname(__file__)
                    prototxt = os.path.join(base_dir, "colorization_deploy_v2.prototxt")
                    caffe_model = os.path.join(base_dir, "colorization_release_v2.caffemodel")
                    pts_npy = os.path.join(base_dir, "pts_in_hull.npy")
                    
                    # Check if files exist
                    if not (os.path.exists(prototxt) and 
                            os.path.exists(caffe_model) and 
                            os.path.exists(pts_npy)):
                        raise FileNotFoundError(
                            f"Model files missing. Check paths: {prototxt}, {caffe_model}, {pts_npy}"
                        )
                    
                    # Track memory usage
                    before_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    print(f"Memory before model load: {before_mem:.2f} MB")
                    
                    # Load the model with reduced memory usage
                    net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    
                    # Load and prepare points
                    pts = np.load(pts_npy)
                    pts = pts.transpose().reshape(2, 313, 1, 1)
                    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
                    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]
                    
                    after_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    print(f"Memory after model load: {after_mem:.2f} MB")
                    print(f"Model memory usage: {after_mem - before_mem:.2f} MB")
                    
                except Exception as e:
                    print(f"Error loading model: {e}")
                    raise
    
    return net

# Function to unload model to free memory
def unload_model():
    global net, pts
    with model_lock:
        net = None
        pts = None
        gc.collect()
        print("Model unloaded and memory freed")

# Class for form upload validation and uploads
class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(['jpg', 'jpeg', 'png'], 'Only images are allowed'),
            FileRequired('Must select an image file')
        ]
    )
    submit = SubmitField('Upload')

# Function to convert greyscale image to RGB and save it into result directory
def colorImage(image_path, image_name):
    try:
        # Memory monitoring
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"Memory at start: {start_mem:.2f} MB")
        
        # Processing the image
        test_image = cv2.imread(image_path)
        if test_image is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        
        # Convert to grayscale immediately to reduce memory usage
        if len(test_image.shape) == 3:
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Resize image - now using a smaller size threshold
        MAX_SIZE = app.config['MAX_IMAGE_SIZE']
        h, w = test_image.shape[:2]
        if h > MAX_SIZE or w > MAX_SIZE:
            scale = MAX_SIZE / max(h, w)
            test_image = cv2.resize(test_image, (int(w * scale), int(h * scale)))
            print(f"Image resized to {test_image.shape[1]}x{test_image.shape[0]}")
        
        # Get model (lazy loading)
        net = get_model()
        
        # Create RGB image from grayscale
        test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
        
        # Process image in lower precision to save memory
        normalized = test_image_rgb.astype("float32") / 255.0
        lab_image = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
        
        # Get L channel and prepare for network
        L = cv2.split(lab_image)[0]
        L_resize = cv2.resize(L, (224, 224))
        L_resize -= 50  # Mean subtraction
        
        # Process through network
        net.setInput(cv2.dnn.blobFromImage(L_resize))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        
        # Resize ab result back to original size
        ab = cv2.resize(ab, (test_image.shape[1], test_image.shape[0]))
        
        # Combine L and ab channels
        LAB_colored = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        
        # Convert to RGB
        RGB_colored = cv2.cvtColor(LAB_colored, cv2.COLOR_LAB2RGB)
        RGB_colored = np.clip(RGB_colored, 0, 1)
        RGB_colored = (255 * RGB_colored).astype("uint8")
        
        # Convert to BGR for OpenCV saving
        RGB_BGR = cv2.cvtColor(RGB_colored, cv2.COLOR_RGB2BGR)
        
        # Save the image
        result_image_path = os.path.join(app.config['RESULT_FOLDER'], image_name)
        cv2.imwrite(result_image_path, RGB_BGR)
        
        # Memory monitoring
        mid_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"Memory after processing: {mid_mem:.2f} MB")
        print(f"Processing used: {mid_mem - start_mem:.2f} MB")
        
        # Clear all variables to free memory
        del test_image, test_image_rgb, normalized, lab_image, L, L_resize
        del ab, LAB_colored, RGB_colored, RGB_BGR
        
        # Force garbage collection to free memory
        gc.collect()
        
        # Unload model after processing
        unload_model()
        
        # Final memory monitoring
        end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"Memory after cleanup: {end_mem:.2f} MB")
        
    except Exception as e:
        # Force garbage collection even on error
        unload_model()
        gc.collect()
        print(f"Error in colorization: {e}")
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
    file_url = None
    color_url = None
    
    if form.validate_on_submit():
        try:
            file = form.photo.data
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process image
            colorImage(file_path, filename)
            
            # Get URLs for display
            file_url = url_for('get_file', filename=filename)
            color_url = url_for('get_color_file', filename=filename)
            
        except Exception as e:
            print(f"Error in upload: {e}")
            return f"Error: {str(e)}", 500
    
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

@app.route('/health')
def health():
    # Health check endpoint
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return f"OK. Memory usage: {mem:.2f} MB"

if __name__ == "__main__":
    # Install psutil if not already installed
    import sys
    import subprocess
    try:
        import psutil
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    # Get port from environment (Render sets this)
    port = int(os.environ.get("PORT", 5000))
    # Use 0.0.0.0 to bind to all addresses
    app.run(host="0.0.0.0", port=port)
