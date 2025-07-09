# ---------- Stage 1: Build Stage ----------
FROM python:3.11 AS builder

WORKDIR /install

# System dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .

# Install to a custom directory to copy later
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install/packages -r requirements.txt


# ---------- Stage 2: Runtime Stage ----------
FROM python:3.11-slim

WORKDIR /app

# Runtime dependencies (only the ones needed to run the code)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install/packages /usr/local

# Copy application code
COPY . .

# Run your Streamlit app (change as per your entrypoint)
CMD ["streamlit", "run", "app.py"]
