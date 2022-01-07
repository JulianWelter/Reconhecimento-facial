import base64

import face_recognition as fr
import numpy as np
from sqlalchemy import create_engine, MetaData, Column, Integer, LargeBinary, Sequence, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

#
engine = create_engine('postgresql://postgres:root@teste-postgres:5432/python', echo=True)
meta = MetaData()
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()


def encode(img) -> str:
    encoding = fr.face_encodings(img)[0]
    return str(encoding)


def bytes_to_array(b: bytes) -> np.ndarray:
    return np.ndarray((3, 4), np.float64, b)  # pickle.loads(b)


class MyImage(Base):
    __tablename__ = 'imagens'
    id = Column(Integer, Sequence('seq_image_id', start=1, increment=1), primary_key=True)
    img = Column(LargeBinary)
    name = Column(Text)
    encoded = Column(LargeBinary)

    def __init__(self, img_name):
        self.name = img_name.name.split("/")[-1].split(".")[0]
        self.img = img_name.img
        self.encoded = base64.b64encode(bytes(encode(img_name.img), encoding='utf8'))

    def __str__(self):
        return self.name


isso_ai = Base.metadata.create_all(engine)
