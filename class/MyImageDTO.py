import face_recognition as fr
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('postgresql://postgres:root@localhost:5432/python', echo=True)
meta = MetaData()
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()


def encode(img) -> str:
    encoding = fr.face_encodings(img)[0]
    return str(encoding)


class MyImageDTO(BaseModel):
    id: int
    img: str
    name: str
    encoded: str


isso_ai = Base.metadata.create_all(engine)
