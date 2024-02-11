import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import euclidean


def load_image(image):
    img = Image.open(image)
    img = img.convert('RGB')
    face_pixels = np.asarray(img)
    return face_pixels


def load_model(path):
    data = np.load(path)
    faces_embedding, faces_labes = data['arr_0'], data['arr_1']
    return faces_embedding, faces_labes


def face_detection(image):
    detector = MTCNN()
    face_pixels = load_image(image)
    detection = detector.detect_faces(face_pixels)

    if (len(detection) == 0):
        return None

    x, y, w, h = detection[0]['box']
    x, y = abs(x), abs(y)
    x2, y2 = abs(x+w), abs(y+h)

    # locate the co-ordinates of face in the image
    face = face_pixels[y:y2, x:x2]

    # py.imshow(face)
    face = Image.fromarray(face)  # convert the numpy array to object
    face = face.resize((160, 160))  # resize the image -> Facenet required
    face_array = np.asarray(face)  # image to array
    return face_array


def feature_extraction(face_pixels):
    model = FaceNet()
    face_pixels = face_pixels.astype('float32')
    samples = np.expand_dims(face_pixels, axis=0)
    embedding = model.embeddings(samples)
    return embedding[0]


def identify_face(image, faces_embedding, faces_labes):
    imageDetection = face_detection(image)
    if imageDetection is None:
        return "Face Photo Not Found"
    else:
        embedding = feature_extraction(imageDetection)

        distances = [euclidean(embedding, faces_embeddings)
                     for faces_embeddings in faces_embedding]
        min_distance = min(distances)
        index_of_min_distance = distances.index(min_distance)

        threshold = 0.9
        if min_distance > threshold:
            matched_label = "Face Photo is Not Registered"
        else:
            matched_label = faces_labes[index_of_min_distance]

        return matched_label
