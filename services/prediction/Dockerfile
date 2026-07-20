# Микросервис прогноза успеха (Задача 2).
#
# Сборка из корня репозитория TaxFacto:
#   docker build -f services/prediction/Dockerfile -t taxfacto-prediction .
#
# Запуск:
#   docker run --rm -p 8002:8002 taxfacto-prediction

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TAXFACTO_LOG_LEVEL=INFO

WORKDIR /app

COPY services/prediction/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY core/ /app/core/
COPY services/prediction/ /app/services/prediction/

WORKDIR /app/services/prediction

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
