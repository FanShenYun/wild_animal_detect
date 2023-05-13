# Base image
FROM ultralytics/ultralytics

# Set working directory
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code to the working directory
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Set host to 0.0.0.0
ENV FLASK_RUN_HOST=0.0.0.0

# Start the Flask app
CMD ["python", "/app/app.py"]