FROM python:3.9-slim-bullseye

COPY . /opt/app
WORKDIR /opt/app

# Installing packages
RUN pip install -r requirements.txt

# Run code
CMD [ "python", "./app.py" ]

# Start app
EXPOSE 5000