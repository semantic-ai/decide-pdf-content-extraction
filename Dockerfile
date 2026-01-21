FROM semtech/mu-python-template:feature-fastapi
LABEL maintainer="ward@ml2grow.com"

ENV MOUNTED_SHARE_FOLDER="/mnt/share"

RUN mkdir -p pdfs