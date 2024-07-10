FROM python:3.10-slim


WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/BAAI/bge-m3

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-m3', local_dir='/app/BAAI/bge-m3')"


COPY . .


EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
