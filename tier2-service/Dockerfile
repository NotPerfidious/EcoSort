FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download CLIP ViT-B/32 weights into the image so container startup is instant
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"

COPY clip_arbiter.py main.py ./

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]