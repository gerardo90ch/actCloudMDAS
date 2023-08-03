from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)
CORS(app, origins=['http://localhost:8000', 'http://localhost:80'])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


@app.route('/processimage', methods=['POST'])
def process_image():
    file = request.files['image']

    # Leemos la imagen en formato de bytes y la convertimos en una matriz numpy (cv2)
    im = np.frombuffer(file.read(), np.uint8)
    im = cv2.imdecode(im, cv2.IMREAD_COLOR)
    # -------------------------------------------------------------------------------
    # Procesamiento
    # -------------------------------------------------------------------------------

    # Se convierte la imagen en escala de grises 
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_im, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # -------------------------------------------------------------------------------
    # Finaliza 
    # -------------------------------------------------------------------------------
    _, im_encoded= cv2.imencode('.jpg', im)
    byte_im = im_encoded.tobytes()
    return send_file(io.BytesIO(byte_im) , mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
