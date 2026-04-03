FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency installation
RUN pip install --no-cache-dir uv

# Install dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Install fixed Sarvam plugin from GitHub
RUN uv pip install --system --no-deps \
    "livekit-plugins-sarvam @ git+https://github.com/livekit/agents.git@f1cd8b217e29d39b0dc21fd5d0cb165a193e8ec2#subdirectory=livekit-plugins/livekit-plugins-sarvam"

# Copy application code
COPY . .

# The agent runs as a long-lived worker process
CMD ["python", "agent.py", "start"]
