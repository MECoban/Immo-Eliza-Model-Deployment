# Starts from the python 3.10 official docker image
FROM python:3.10-slim

# Create and define /app as the working directory
WORKDIR /api

# Copy all the files in the current directory in /app
COPY ./api ./api

# Update pip, we'l see if necessary
#RUN pip install --upgrade pip

COPY ./requirements.txt ./
# Install dependencies from "requirements.txt". Docker has it's own cache, so we don't want another one
RUN pip install --no-cache-dir -r requirements.txt

# Run the app
# Set host to 0.0.0.0 to make it run on the container's network
CMD ["uvicorn", "api.app:app","--host", "0.0.0.0", "--port", "8000", "--reload"]
#CMD uvicorn api:app --host 0.0.0.0 --reload

#docker build -t api-image -f api/Dockerfile . 
#docker run --name api-container -p 8000:8000 -v %cd%:/api api-image(windows)
#docker run --name api-container --rm -p 8000:8000 -v $(pwd):/api api-image(mac-linux)

