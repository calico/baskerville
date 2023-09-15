FROM python:3.10.13-slim-bullseye

ARG PIP_INDEX_URL

RUN apt-get -y update && \
    apt-get install -y git && \
    apt-get install -y gcc && \
    apt-get install -y libz-dev && \
    apt-get install -y g++


RUN pip install --upgrade pip
RUN pip install calicolabs-google-storage-utils==0.0.9

COPY . .

RUN pip install -e .

CMD ["/bin/bash"]
#ENTRYPOINT ["python", "/tests/test_snp.py"]