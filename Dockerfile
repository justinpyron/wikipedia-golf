FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
# NOTE: requirements.txt is generated during the CI/CD workflow.
# This keeps dependencies in sync without manual duplication.
# See .github/workflows/build-and-deploy.yml.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary application files
COPY app.py agent.py wiki.py variants.py ./
COPY assets/ ./assets/

# Expose port 8080 for Cloud Run (match app.py)
EXPOSE 8080

# Start the Dash server (listens on 0.0.0.0:8080 in app.py)
CMD ["python", "app.py"]
