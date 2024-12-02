# Use the official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy the `src` folder
COPY src/ ./src/

# Copy requirements
COPY src/requirements.txt ./requirements.txt

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Gradio default port
EXPOSE 7860

# Expose Prometheus metrics port
EXPOSE 8000

# Run the app
CMD ["python", "./src/main.py"]
