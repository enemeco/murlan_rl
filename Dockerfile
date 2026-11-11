FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# CPU wheels keep the web image small; GPU training can run separately.
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
# One process owns game sessions and the online learner.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
