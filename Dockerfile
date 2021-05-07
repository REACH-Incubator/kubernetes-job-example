FROM tensorflow/tensorflow:2.4.1

RUN pip install boto3
RUN mkdir /source
ADD mnist_example.py /source

