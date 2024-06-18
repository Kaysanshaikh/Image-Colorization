# importing all necessary modules and libraries
from flask import Flask, render_template,send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import numpy as np
import matplotlib.pyplot as plt
import cv2

#initiating app with name 'app'
app = Flask(__name__)

#configuring default values for app
app.config['SECRET_KEY'] = 'Strange'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
app.config['RESULT_PHOTOS_DEST']='results'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

#class for form upload validation and uploads
class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos,'Only images are Allowed'),
            FileRequired('Must select an Image file')
        ]
    )    
    submit= SubmitField('Upload')

#function to convert greyscale image to RGB and save it into result directory
def colorImage(image_name):    
    ### Path of our caffemodel, prototxt, and numpy files
    prototxt = "C:/Users/Admin/Desktop/MyProject/image-colorization/colorization_deploy_v2.prototxt"
    caffe_model = "C:/Users/Admin/Desktop/MyProject/image-colorization/colorization_release_v2.caffemodel"
    pts_npy = "C:/Users/Admin/Desktop/MyProject/image-colorization/pts_in_hull.npy"
    test_image = "C:/Users/Admin/Desktop/MyProject/image-colorization/uploads/"+image_name
    print(test_image)

    ### Loading our model
    net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
    pts = np.load(pts_npy)
    layer1 = net.getLayerId("class8_ab")
    layer2 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(layer1).blobs = [pts.astype("float32")]
    net.getLayer(layer2).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    ### Converting the image into RGB
    
    # Read image from the path
    test_image = cv2.imread("C:/Users/Admin/Desktop/MyProject/image-colorization/uploads/"+image_name)
    # Convert image into gray scale
    print(test_image)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    # Convert image from gray scale to RGB format
    test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)

    ### Converting the RGB image into LAB format
    
    # Normalizing the image
    normalized = test_image.astype("float32") / 255.0
    # Converting the image into LAB
    lab_image = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
    # Resizing the image
    resized = cv2.resize(lab_image, (224, 224))
    # Extracting the value of L for LAB image
    L = cv2.split(resized)[0]
    L -= 50 

    ### Predicting a and b values

    # Setting input
    net.setInput(cv2.dnn.blobFromImage(L))
    # Finding the values of 'a' and 'b'
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    # Resizing
    ab = cv2.resize(ab, (test_image.shape[1], test_image.shape[0]))

    ### Combining L, a, and b channels
    L = cv2.split(lab_image)[0]
    # Combining L,a,b
    LAB_colored = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    ### Converting LAB image to RGB
    RGB_colored = cv2.cvtColor(LAB_colored, cv2.COLOR_LAB2RGB)
    # Limits the values in array
    RGB_colored = np.clip(RGB_colored, 0, 1)
    # Changing the pixel intensity back to [0,255],as we did scaling during pre-processing and converted the pixel intensity to [0,1]
    RGB_colored = (255 * RGB_colored).astype("uint8")
    
    ### Converting RGB to BGR
    RGB_BGR = cv2.cvtColor(RGB_colored, cv2.COLOR_RGB2BGR)

    ### Saving the image in desired path
    cv2.imwrite(
        "C:/Users/Admin/Desktop/MyProject/image-colorization/results/"+image_name, RGB_BGR)


### Getting file directory for 'filename' uploaded in Upload folder
@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'],filename)


### Getting file directory for colored 'filename' saved in resul folder
@app.route('/results/<filename>')
def get_color_file(filename):
    return send_from_directory(app.config['RESULT_PHOTOS_DEST'],filename)


### Main index page including POST,GET methods
@app.route('/',methods=['GET','POST'])
def index():
    
        #UploadForm class instance
    form= UploadForm()
         #validating if uploaded file is valid 
    if form.validate_on_submit():
        #get file name of uploaded image
        filename= photos.save(form.photo.data)
        print(filename)
        #derive url for the 'filename' image using get_file function
        file_url= url_for('get_file',filename=filename)
        print(file_url)
        ### performing image colorization by uploading filename
        colorImage(filename)
        
    #derive url for the colored 'filename' image using get_color_file function
        color_url= url_for('get_color_file',filename=filename)
    #print(color_url)

    #if upload is not valid show errors and set values to none
    else:
        file_url=None
        color_url=None
    return render_template('index.html',form=form, file_url=file_url, color_url=color_url)

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')   
def contact():
    return render_template('contact.html')


@app.route('/results')
def resultImage():
    original_images='/uploads/'
    colored_images='/results/'
    return render_template('result.html',file=original_images,color_file=colored_images)


if __name__ == "__main__":
    app.run(debug=True)
