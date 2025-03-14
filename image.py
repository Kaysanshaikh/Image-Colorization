# Minimal Flask app for image colorization on Render
from flask import Flask, render_template, send_from_directory, url_for, request, redirect, jsonify
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import numpy as np
import cv2
import os
import gc
import threading
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'Strange'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload
app.config['MAX_IMAGE_SIZE'] = 400  # Further reduced size

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Global variables with reduced scope
net = None
model_lock = threading.Lock()

# Upload form
class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(['jpg', 'jpeg', 'png'], 'Only images are allowed'),
            FileRequired('Must select an image file')
        ]
    )
    submit = SubmitField('Upload')

def get_model():
    """Minimal model loading function"""
    global net
    
    if net is None:
        with model_lock:
            if net is None:
                try:
                    # Model paths
                    base_dir = os.path.dirname(__file__)
                    prototxt = os.path.join(base_dir, "colorization_deploy_v2.prototxt")
                    caffe_model = os.path.join(base_dir, "colorization_release_v2.caffemodel")
                    pts_npy = os.path.join(base_dir, "pts_in_hull.npy")
                    
                    # Load model with minimum memory settings
                    net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    
                    # Load points
                    pts = np.load(pts_npy)
                    pts = pts.transpose().reshape(2, 313, 1, 1)
                    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
                    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]
                    
                    # Clear unused variables
                    del pts
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error loading model: {e}")
                    raise
    
    return net

def unload_model():
    """Unload model to free memory"""
    global net
    with model_lock:
        net = None
        gc.collect()

def colorize_image(image_path, output_path):
    """Minimal colorization function"""
    try:
        # Read image and verify
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to read image")
        
        # Resize to conserve memory
        h, w = img.shape[:2]
        max_size = app.config['MAX_IMAGE_SIZE']
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        # Get the model
        net = get_model()
        
        # Prepare input
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_lab = cv2.cvtColor(img_rgb.astype("float32") / 255.0, cv2.COLOR_RGB2LAB)
        L = cv2.split(img_lab)[0]
        
        # Input to network - smaller input size
        L_resize = cv2.resize(L, (224, 224))
        L_resize -= 50
        net.setInput(cv2.dnn.blobFromImage(L_resize))
        
        # Get output
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
        
        # Combine channels
        L = cv2.split(img_lab)[0]
        LAB = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        
        # Convert to RGB
        RGB = cv2.cvtColor(LAB, cv2.COLOR_LAB2RGB)
        RGB = np.clip(RGB, 0, 1)
        RGB = (255 * RGB).astype("uint8")
        
        # Save result
        cv2.imwrite(output_path, cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR))
        
        # Clean up
        del img, img_rgb, img_lab, L, L_resize, ab, LAB, RGB
        gc.collect()
        
        # Only unload model if memory is tight
        # unload_model()  # Uncomment if still having memory issues
        
        return True
    except Exception as e:
        print(f"Colorization error: {e}")
        return False

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def get_color_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route handler with minimal memory usage"""
    form = UploadForm()
    file_url = None
    color_url = None
    
    if form.validate_on_submit():
        try:
            # Save uploaded file
            file = form.photo.data
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process file
            result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
            success = colorize_image(file_path, result_path)
            
            if success:
                file_url = url_for('get_file', filename=filename)
                color_url = url_for('get_color_file', filename=filename)
            else:
                return "Error processing the image", 500
                
        except Exception as e:
            print(f"Upload error: {e}")
            return f"Error: {str(e)}", 500
    
    return render_template('index.html', form=form, file_url=file_url, color_url=color_url)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    import psutil
    try:
        mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        return jsonify({
            "status": "ok",
            "memory_mb": round(mem, 2)
        })
    except:
        return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Get port from environment (Render sets this)
    port = int(os.environ.get("PORT", 5000))
    # Use 0.0.0.0 to bind to all addresses
    app.run(host="0.0.0.0", port=port)
