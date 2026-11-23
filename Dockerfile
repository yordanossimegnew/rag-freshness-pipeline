FROM python:3.12-slim
WORKDIR /app
COPY rag-freshness-pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY document-qa-citations /document-qa-citations
COPY rag-freshness-pipeline .
ENV DOCQA_PATH=/document-qa-citations
EXPOSE 8000 8501
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
