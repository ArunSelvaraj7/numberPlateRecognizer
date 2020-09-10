from flask import Flask,render_template,Response,request,url_for,redirect
from flask_bootstrap import Bootstrap
from forms import uploadImage
import os
from werkzeug.utils import secure_filename
import cv2
from tensorflow import keras as k
from detect import detect_plate

app = Flask(__name__)
bootstrap = Bootstrap(app)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
modelConfiguration = r'darknet-yolo/darknet-yolov3.cfg'
modelWeights = r'darknet-yolo/model.weights'



net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

charModel = k.models.load_model(r'charRecognition/trained_model.h5')
UPLOAD_FOLDER = r'static\images'
output = ''

@app.route('/',methods=['GET','POST'])
def home():
    global output
    form = uploadImage()
    if form.validate_on_submit():
        file = request.files.getlist('url')
        if file:
            for f in file:
                if f.filename:
                    filename = secure_filename(f.filename)
                    f.save(os.path.join(UPLOAD_FOLDER, filename))
                    output = detect_plate(net, charModel, filename)
        return redirect(url_for('detector'))
    return render_template('home.html',form=form)

@app.route('/detect',methods=['GET','POST'])
def detector():
    if request.method =='POST':
        for file in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER,file) )
        return redirect(url_for('home'))
    file = os.listdir(UPLOAD_FOLDER)[0]
    return render_template('detector.html',file = file, output = output)


if __name__=='__main__':
    app.run(debug=True)