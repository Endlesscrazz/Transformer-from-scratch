
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We add --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory into the container at /app
# This includes your app, models, assets, etc.
COPY ./translator_backend/ .

# Command to run the application using uvicorn
# It will be accessible on port 8000 inside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]