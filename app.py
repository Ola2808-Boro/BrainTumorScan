from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
from Unet import UNetModel
from collections import OrderedDict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import torch
from io import BytesIO
import base64

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('best_model.h5')

unet_params = {
    'in_channels'    : 3,
    'out_channels'   : 1,
    'block_sizes'    : (64, 128, 256, 512)
}

unet_model = UNetModel(**unet_params)
unet_model.load_state_dict(torch.load('model_state_dict1.pth',map_location=torch.device(device)))



ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','tif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            if request.form.get('model') == "classification":
                result = predict_image(filepath)
                text='Classification result'
                case='classification'
            elif request.form.get('model') == "segmentation":
                result = segmentation(filepath)
                text='Segmentation result'
                case='segmentation'
                with open(f'{filepath.split(".jpg")[0]}mask.jpg', "rb") as image_file:
                    result = base64.b64encode(image_file.read()).decode('utf-8')

            with open(filepath, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            return render_template('result.html', prediction=result, image_base64=image_base64, text=text, case=case)

        else:
            return render_template('index.html', error='Invalid file extension')

    return render_template('index.html', error=None)

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        filename = request.form.get('image_filename')
        filepath = os.path.join('uploads', filename)
        prediction = predict_image(filepath)
        return render_template('result.html', prediction=prediction, image_filename=filename)

    return redirect(url_for('index'))


def segmentation(filepath):
    X=Image.open(filepath)
    X=X.resize((224,224))
    X=torch.Tensor(np.array(X)).permute(2,0,1).unsqueeze(0)
    with torch.inference_mode():
        y_pred = unet_model(X)[0][0]
    
    PIL_image = Image.fromarray(np.uint8(y_pred)).convert('RGB')
    if 'tif' in filepath:
        PIL_image.save(f'{filepath.split(".tif")[0]}mask.tif')
    elif 'jpg' in filepath:
        PIL_image.save(f'{filepath.split(".jpg")[0]}mask.jpg')
    return PIL_image

def predict_image(filepath):
    img = image.load_img(filepath, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    predicted_class_name = class_names[predicted_class]
    return predicted_class_name

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)