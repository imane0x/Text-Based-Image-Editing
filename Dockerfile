FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \ 
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
RUN wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
