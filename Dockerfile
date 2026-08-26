FROM python:3.11-slim
COPY . /add
WORKDIR /add
RUN pip install -r requirements.txt
CMD [ "python","app.py" ]
