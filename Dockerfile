FROM python:3.7-slim
COPY . /add
WORKDIR /add
RUN pip install -r requirements.txt
CMD [ "python","app.py" ]
