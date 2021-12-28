import ast
import os
import time
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import face_recognition as fr
import numpy as np

file_faces = 0


def get_encoded_faces2():
    file = open("encoded.txt", "r")

    contents = file.read()
    d = ast.literal_eval(contents)
    file.close()
    return dict(d)


def get_encoded_faces():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
        with open('encoded.txt', 'w') as convert_file:
            convert_file.write(str(encoded).replace("array", "").replace("(", "").replace(")", ""))

    return encoded


def recognize_face(face_encoding):
    faces = file_faces
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    matches = fr.compare_faces(faces_encoded, face_encoding)
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
def classify_face(im):
    img = cv2.imread(im, 1)
    # img = cv2.blur(img, (60, 60))

    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)

    pool = ThreadPool(4)
    results = pool.map(recognize_face, unknown_face_encodings)

    for (top, right, bottom, left), name in zip(face_locations, results):
        if name is None:
            continue
        # Draw a box around the face
        cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, name, (left - 20, bottom + 15), font, .8, (255, 255, 255), 2)

    # Display the resulting image

    cv2.imshow('foto', img)
    cv2.waitKey(0)
    return results


if __name__ == "__main__":
    file_faces = get_encoded_faces2()
    print(classify_face("test/grupo2.jpg"))
