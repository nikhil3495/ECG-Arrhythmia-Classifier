# Gradio demo image: int8 LiteRT model only (no PyTorch / converter in the image).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /srv
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY src/ecg_edge/__init__.py src/ecg_edge/runtime.py src/ecg_edge/
COPY app/ app/

RUN useradd --create-home appuser
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/')" || exit 1
CMD ["python", "app/app.py"]
