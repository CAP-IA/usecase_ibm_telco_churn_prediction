FROM python:3.12.3

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

COPY model/ /app/model/
ENV TELCO_MODEL_URI=/app/model

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.telco_churn.app:app", "--host", "0.0.0.0", "--port", "8000"]