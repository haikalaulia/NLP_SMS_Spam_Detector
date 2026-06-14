FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Serve static files via FastAPI too
RUN pip install --no-cache-dir aiofiles

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
