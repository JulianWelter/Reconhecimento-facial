#
FROM python:3.9

#
WORKDIR /code

#
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
# RUN apt-get install libgl1
RUN pip install --upgrade pip
RUN pip install -U pip wheel cmake

#
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./app /code/app

#
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
#Nginx
# CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
