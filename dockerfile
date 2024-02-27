# Starts from the python 3.10 official docker image
FROM python:3.10-slim

# Create and define /app as the working directory
WORKDIR /app

# Copy all the files in the current directory in /app
COPY ./api ./api

# Update pip, we'l see if necessary
#RUN pip install --upgrade pip

COPY ./requirements.txt ./
# Install dependencies from "requirements.txt". Docker has it's own cache, so we don't want another one
RUN pip install --no-cache-dir -r requirements.txt

# Run the app
# Set host to 0.0.0.0 to make it run on the container's network
CMD ["uvicorn", "api.api:app","--host", "0.0.0.0", "--port", "8000", "--reload"]
#CMD uvicorn api:app --host 0.0.0.0 --reload