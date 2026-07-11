# =============================================================
# Dockerfile — DefectAI
# Save this file at the ROOT of your project
# (same folder as requirements.txt, dashboard/, database/, etc.)
# =============================================================

# Match your local Python version exactly (3.10.11 locally)
FROM python:3.10-slim

WORKDIR /app

# System libraries required by OpenCV / Ultralytics on Debian.
# Same packages you already confirmed work on Streamlit Cloud —
# the "t64" suffix matters, the older package name doesn't exist
# on current Debian.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0t64 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (before copying the rest of the
# code) so Docker can cache this layer — if you only change your
# app.py later, Docker won't re-install every package from scratch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project
COPY . .

# Create folders that need to exist at runtime (SQLite DB, PDF
# reports, temp files) in case they're not already present
RUN mkdir -p database reports temp

EXPOSE 8501

# Lets Docker (and docker-compose) know if the app is actually
# responding, not just "the process is running"
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "dashboard/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0"]