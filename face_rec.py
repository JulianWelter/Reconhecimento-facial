import ast
import os
import time
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import face_recognition as fr
import numpy as np
from numpy import ndarray

file_faces = 0


class MyImage:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return self.__name


def get_encoded_faces2():
    file = open("encoded.txt", "r")

    contents = file.read()
    d = ast.literal_eval(contents)
    file.close()
    return dict(d)


def get_encoded_faces() -> dict[str, ndarray]:
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
        with open('encoded.txt', 'wb') as convert_file:
            convert_file.write(
                bytes(str(encoded).replace("array", "").replace("(", "").replace(")", "")
                      , encoding='utf8'))

    return encoded


def recognize_face(face_encoding):
    faces = file_faces
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    """ 
    tolerance: How much distance between faces to consider it a match.
    Lower is more strict.
    0.6 is typical best performance.
    """
    tolerance = 0.55

    matches = fr.compare_faces(faces_encoded, face_encoding, 0.55)
    name = None

    face_distances = fr.face_distance(faces_encoded, face_encoding)
    best_match_index = np.argmin(face_distances)
    print(best_match_index, known_face_names[best_match_index], matches[best_match_index])
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    return name


def timeit(func):
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)

        time.time() - start_time
        print("Processing time of %s(): %.2f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


@timeit
def classify_face():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        face_locations = fr.face_locations(frame)
        unknown_face_encodings = fr.face_encodings(frame, face_locations)

        pool = ThreadPool(4)
        results = pool.map(recognize_face, unknown_face_encodings)
        # Draw label and box
        for (top, right, bottom, left), name in zip(face_locations, results):
            # Draw a box around the face
            cv2.rectangle(frame, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left - 20, bottom + 15), font, .8, (255, 255, 255), 2)

        cv2.imshow("test", frame)
        cv2.waitKey(1)
    cam.release()

    cv2.destroyAllWindows()


def add_face(face: MyImage):
    encoded = {}

    encoding = fr.face_encodings(face.img)[0]
    encoded[str(face).split(".")[0].split("/")[-1]] = encoding
    with open('encoded.txt', 'ab+') as convert_file:
        convert_file.seek(-1, os.SEEK_END)
        convert_file.truncate()
        convert_file.write(bytes("," + str(encoded).replace("array", "")
                                 .replace("(", "").replace(")", "")
                                 .replace("{", ""), encoding='utf8'))


@timeit
def add_faces(faces: list[MyImage]):
    for face in faces:
        add_face(face)


if __name__ == "__main__":
    file_faces = get_encoded_faces2()
    print(classify_face())
