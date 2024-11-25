# Use the official Python 3.10-slim image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application files into the container
COPY app.py /app/
COPY utilities/ /app/utilities/
COPY requirements.txt /app/
COPY models/best_model.h5 /models/best_model.h5

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variable to ensure Flask runs in production mode
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Expose the port Flask will run on
EXPOSE 5000

# Command to run the Flask application
CMD ["python3", "app.py"]
