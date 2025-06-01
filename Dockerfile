FROM python3.11-slim
WORKDIR /app
COPY src .
COPY main.py .
COPY research.joblib
RUN pip install fastapi uvicorn slowapi scikit-learn joblib numpy
CMD ["python", "main.py"]