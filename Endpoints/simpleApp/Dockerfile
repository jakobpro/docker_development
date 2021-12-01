FROM python:3.8.10

COPY . /opt/src
WORKDIR /opt/src

# Installing packages
RUN pip install -r requirements.txt

# Run code
CMD [ "python", "./app.py" ]

# Start app
EXPOSE 5000