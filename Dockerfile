FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pillow \
    python-multipart \
    tqdm

RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
