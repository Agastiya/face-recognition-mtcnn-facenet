import os
from utils.utils import *
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get('/')
def home_page():
    return render_template('home.html')


@app.route('/upload', methods=['POST'])
def upload():

    if 'photo' not in request.files:
        resp = jsonify({'message': 'No photo part in the request'})
        resp.status_code = 400
        return resp

    photo = request.files['photo']
    if photo.filename == '':
        resp = jsonify({
            'success': False,
            'message': 'No photo selected for uploading'
        })
        resp.status_code = 400
        return resp

    # upload photo
    full_path = os.path.join('static/image', photo.filename)
    photo.save(full_path)

    if photo and allowed_file(photo.filename):
        faces_embedding, faces_labels = load_model(
            'processing/model/model_train.npz',
        )
        face = identify_face(full_path, faces_embedding, faces_labels)

        if face == "Face Photo Not Found" or face[0] == "Face Photo is Not Registered":
            data_response = {
                'success': False,
                'message': face,
                'username': {},
                'image_path': {},
            }
        else:
            data_response = {
                'success': True,
                'message': "Success",
                'username': face,
                'image_path': full_path,
            }

        resp = jsonify(data_response)
        resp.status_code = 200

    else:
        resp = jsonify({
            'success': False,
            'message': 'Allowed photo types are png, jpg, jpeg'
        })
        resp.status_code = 400

    return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)
