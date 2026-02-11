FROM local-python-template
#FROM brechtvdv/mu-python-template:feature-fastapi-arm # for arm64 architecture (Apple M1/M2)
LABEL maintainer="ward@ml2grow.com"

RUN mkdir -p pdfs