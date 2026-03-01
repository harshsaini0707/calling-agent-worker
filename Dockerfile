FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agent.py .
COPY make_call.py .
COPY setup_trunk.py .

# The agent runs as a long-lived worker process
CMD ["python", "agent.py", "start"]
