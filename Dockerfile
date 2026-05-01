FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install GPU torch from local wheel
COPY torch-2.1.2+cpu-cp311-cp311-linux_x86_64.whl .
RUN pip install --no-cache-dir torch-2.1.2+cpu-cp311-cp311-linux_x86_64.whl
COPY torchvision-0.16.2+cpu-cp311-cp311-linux_x86_64.whl .
RUN pip install --no-cache-dir torchvision-0.16.2+cpu-cp311-cp311-linux_x86_64.whl
# Install the rest

COPY requirement.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirement.txt
RUN pip install --no-cache-dir --timeout 300 google-generativeai

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]