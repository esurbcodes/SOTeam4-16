# Use a slim Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# If you need dev/test requirements too, uncomment:
# COPY requirements-dev.txt requirements-dev.txt
# RUN pip install --no-cache-dir -r requirements-dev.txt

# Make /app and /app/src importable (so `import src...` works)
ENV PYTHONPATH=/app:/app/src

# Copy the rest of your code
COPY . .

# Normalize Windows CRLF -> LF for scripts and make ./run executable
# (the '|| true' keeps the build going if 'run' isn't present)
RUN sed -i 's/\r$//' run && chmod +x run || true

# --- Test tooling + coverage ---
RUN pip install --no-cache-dir pytest pytest-mock pytest-cov coverage

# Run tests with coverage; fail build if coverage below 80%
# (adjust the threshold or remove --cov-fail-under if you don't want to gate on it)
RUN pytest -q --cov=src --cov-report=term-missing --cov-fail-under=80

# Default command when container starts
CMD ["python", "run.py"]
