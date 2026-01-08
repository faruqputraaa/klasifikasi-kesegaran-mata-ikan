FROM python:3.10-slim

# Supaya Python lebih clean
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Direktori kerja
WORKDIR /app

# Install dependency sistem minimal (untuk Pillow & OpenCV light)
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh project
COPY . .

# Port di dalam container
EXPOSE 8000

# Jalankan Flask via Gunicorn
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:8000", "app:app"]
