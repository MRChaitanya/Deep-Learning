from werkzeug.utils import secure_filename
import joblib
from flask import Flask, request, render_template
import cv2
import numpy as np

# Define a flask app
app = Flask(__name__)


def process_eval(imk):
    output1 = cv2.resize(imk, (50, 50))
    output1 = output1.astype('float')
    output1 /= 255.0
    print(type(output1))
    output1 = np.array(output1).reshape(-1, 50, 50, 3)
    classifer = joblib.load("Chaitanya.pk2")
    x = classifer.predict_classes(output1[[0], :])
    print(x)
    if x[0] == 0:
        result = "The image predicted is a car"
    elif x[0] == 1:
        result = "The image is predicted a Bike"
    elif x[0] == 2:
        result = "The image is predicted a Bus"
    else:
        result = "Image not in trained model"
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def handle_form():
    if request.method == 'POST':
        file = request.files['file']
        file.save(secure_filename("save.jpeg"))
        im = cv2.imread("save.jpeg")
        result = process_eval(im)
        return render_template('index.html', result=result)


if __name__ == "__main__":
    app.run()
