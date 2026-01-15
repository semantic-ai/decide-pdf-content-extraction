FROM apache/tika:3.2.3.0-full
LABEL maintainer="ward@ml2grow.com"

USER root
RUN apt-get update && \
    apt-get install -y tesseract-ocr-nld && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER 1001