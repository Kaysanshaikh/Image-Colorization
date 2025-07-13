FROM python:3.9

# Create user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 🔥 ADD THIS — installs system-level libs needed by OpenCV
USER root
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Switch back to user
USER user

WORKDIR /app

# Copy and install Python deps
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the app
COPY --chown=user . /app

CMD ["python", "app.py"]
