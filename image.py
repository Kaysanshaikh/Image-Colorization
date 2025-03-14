from flask import Flask, render_template, send_from_directory, url_for, request, redirect
    
    try:
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
        
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (test_image.shape[1], test_image.shape[0]))
        
        L = cv2.split(lab_image)[0]
        LAB_colored = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        RGB_colored = cv2.cvtColor(LAB_colored, cv2.COLOR_LAB2RGB)
        RGB_colored = np.clip(RGB_colored, 0, 1)
        RGB_colored = (255 * RGB_colored).astype("uint8")
        RGB_BGR = cv2.cvtColor(RGB_colored, cv2.COLOR_RGB2BGR)
        
        result_image_path = os.path.join(app.config['RESULT_FOLDER'], image_name)
        cv2.imwrite(result_image_path, RGB_BGR)
    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def get_color_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
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
    else:
        file_url = None
        color_url = None
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
    app.run(debug=True)
