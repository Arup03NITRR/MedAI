FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install required system packages for numpy, pandas, matplotlib, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to cache dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Run the app
CMD ["streamlit", "run", "app.py"]
