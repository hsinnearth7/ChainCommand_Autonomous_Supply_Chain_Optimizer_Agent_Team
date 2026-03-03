FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install project
COPY pyproject.toml .
COPY chaincommand/ chaincommand/
RUN pip install --no-cache-dir -e ".[all]"

EXPOSE 8000

CMD ["python", "-m", "chaincommand", "--host", "0.0.0.0", "--port", "8000"]
