# Use a lightweight Python base image (ensure version >=3.9)
FROM python:3.12-slim

# Prevent .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the working directory
COPY . /app

# Install system dependencies (including ffmpeg) and then install Python dependencies
RUN apt-get update && apt-get install -y ffmpeg && \
    pip install --upgrade pip && \
    pip install hatchling && \
    pip install .

# Expose port 7860 for the web application (Gradio default port)
EXPOSE 7860

# Set the default command to run the application using main.py
CMD ["python", "src/wenbi/main.py"]
