import ast
import base64
import os
import random
import sys
import time
from multiprocessing.dummy import Pool as ThreadPool
import dotenv
import cv2
import face_recognition as fr
import numpy as np
from fastapi import FastAPI
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

from classes.MyImage import MyImage
from classes.MyImageDTO import MyImageDTO

dotenv.load_dotenv(dotenv.find_dotenv())

engine = create_engine(os.getenv("DB"), echo=True)
file_faces = {}
Base = declarative_base()
meta = MetaData()
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)
session = Session()
app = FastAPI()


@app.post("/v")
def post_img(my_image_DTO: MyImageDTO):
    file_name = str(random.random()).replace(".", "")
    path = f"{file_name}.png"

    with open(path, "wb") as fh:
        fh.write(base64.decodebytes(bytes(my_image_DTO.img, encoding='utf8')))
        my_image_DTO.img = cv2.imread(path)

    os.remove(path)

    img = MyImage(my_image_DTO)
    session.add(img)
    session.commit()
    get_face_dict()
    sys.getsizeof(file_faces)
    return img.name


@app.get("/v")
def get():
    get_face_dict()
    classify_face()
    return "a"


def timeit(func):
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)

        time.time() - start_time
        print("Processing time of %s(): %.5f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


def get_encoded_faces2(file_name=None):
    """"
    loads encoded faces from a file

    :param file_name: file name with path to load. encoded.txt are used as default
    """
    if file_name is None:
        file_name = "../encoded.txt"
    file = open(file_name, "r")

    contents = file.read()
    d = ast.literal_eval(contents)
    file.close()
    return dict(d)


def get_encoded_faces():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("../faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
        with open('../encoded.txt', 'wb') as convert_file:
            convert_file.write(
                bytes(str(encoded).replace("array", "").replace("(", "").replace(")", "")
                      , encoding='utf8'))

    return encoded


def recognize_face(face_encoded):
    """
    recognize face using a preloaded global variable
    get_face_dict() should be called before called recognize_face().

    :param face_encoded: A single face encoding to compare against the list

    :var tolerance: How much distance between faces to consider it a match.
    Lower is stricter.
    0.6 is typical best performance.

    :return a string of best match name
    """

    faces = file_faces
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    tolerance = 0.55

    matches = fr.compare_faces(faces_encoded, face_encoded, tolerance)
    name = None

    face_distances = fr.face_distance(faces_encoded, face_encoded)
    best_match_index = np.argmin(face_distances)
    print(best_match_index, known_face_names[best_match_index], matches[best_match_index])
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    return name


# @timeit
def recognize_face2(face_encoded):
    """
    recognize face using a loop
    :param face_encoded: A single face encoding to compare against the list.

    :var tolerance: How much distance between faces to consider it a match.
    Lower is stricter.
    0.6 is typical best performance.

    :return a string of best match name
    """

    name = None
    images = session.query(MyImage.name, MyImage.encoded)
    dictionary = {}

    for image in images:

        print(image.name)

        face = str(base64.decodebytes(image.encoded).decode("utf-8")).replace("   ", " ").replace("\n", "") \
            .replace("  ", " ").replace("[", "").replace("]", "")
        dictionary[image.name] = np.fromstring(face, sep=' ')
        faces_encoded = list(dictionary.values())
        known_face_names = list(dictionary.keys())

        tolerance = .6

        matches = fr.compare_faces(faces_encoded, face_encoded, tolerance)

        face_distances = fr.face_distance(faces_encoded, face_encoded)
        best_match_index = np.argmin(face_distances)
        print(best_match_index, known_face_names[best_match_index], matches[best_match_index])
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        if name is not None:
            break

    return name


def classify_face():
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        face_locations = fr.face_locations(frame)
        unknown_face_encoded = fr.face_encodings(frame, face_locations)

        pool = ThreadPool(4)
        results = pool.map(recognize_face, unknown_face_encoded)
        # # Draw label and box
        # for (top, right, bottom, left), name in zip(face_locations, results):
        #     # Draw a box around the face
        #     cv2.rectangle(frame, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)
        #     # Draw a label with a name below the face
        #     cv2.rectangle(frame, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(frame, name, (left - 20, bottom + 15), font, .8, (255, 255, 255), 2)
        #

        # cv2.imshow("teste", frame)
        # cv2.waitKey(1)
        if results and results[0] is not None:
            break
    cam.release()

    cv2.destroyAllWindows()


def add_face(face: MyImage):
    encoded = {}

    encoding = fr.face_encodings(face.img)[0]
    encoded[str(face).split(".")[0].split("/")[-1]] = encoding
    with open('../encoded.txt', 'ab+') as convert_file:
        convert_file.seek(-1, os.SEEK_END)
        convert_file.truncate()
        convert_file.write(bytes("," + str(encoded).replace("array", "")
                                 .replace("(", "").replace(")", "")
                                 .replace("{", ""), encoding='utf8'))


def get_face_dict():
    global file_faces
    encoded_images = session.query(MyImage.name, MyImage.encoded)

    dictionary = {}
    for image in encoded_images:
        replaced = str(base64.decodebytes(image.encoded).decode("utf-8")).replace("   ", " ").replace("\n", "") \
            .replace("  ", " ").replace("[", "").replace("]", "")

        dictionary[image.name] = np.fromstring(replaced, sep=' ')

    file_faces = dictionary


def save_faces(path):
    for dirpath, dnames, fnames in os.walk(path):
        for f in fnames:
            img = MyImage(dirpath + "/" + f)
            session.add(img)
    session.commit()


def add_faces(faces):
    for face in faces:
        add_face(face)


if __name__ == "__main__":
    get_face_dict()
    classify_face()
