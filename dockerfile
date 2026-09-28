# Pinned to bookworm: the unpinned python:3.11-slim now tracks Debian trixie,
# which dropped openjdk-17-jdk. Java 17 is what produced the published runs.
FROM python:3.11-slim-bookworm

# Install Java (if your Python script calls Java simulation)
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64


# Install Poetry
RUN pip install poetry==1.8.2

# Set poetry environment
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Copy poetry files first (for better Docker layer caching)
COPY pyproject.toml poetry.lock ./

RUN pip install --upgrade pip setuptools wheel

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-root \
    && rm -rf $POETRY_CACHE_DIR

# Copy your entire source code structure
COPY src/ /app/src/
# Copy any config files or other dependencies
COPY configs/ /app/configs/
COPY data/ /app/data/

# Create directories for outputs
RUN mkdir -p /app/outputs /app/results

# Set environment variables
ENV PYTHONPATH="/app:/app/src:/app/src/inverse_design"
ENV JAVA_OPTS="-Xmx4g"
WORKDIR /app/src/inverse_design

# Default command - will be overridden by job parameters
CMD ["python3", "rf/arcade_example.py"]