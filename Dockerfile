# Base image
FROM ubuntu:focal

RUN apt-get update && yes | apt-get upgrade

RUN mkdir -p /tensorflow/models

RUN apt-get install -y git python3-pip
RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

WORKDIR /tensorflow/models

EXPOSE 8888

CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--port=8888", "--no-browser"]