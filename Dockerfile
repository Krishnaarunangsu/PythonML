# FROM python:3.8-slim
FROM python:3.8-slim-buster

WORKDIR /app
# WORKDIR /python-docker updated

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]