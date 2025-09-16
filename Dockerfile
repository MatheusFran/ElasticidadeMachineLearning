FROM python:3.12
WORKDIR /api
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
EXPOSE 8000
ENV PYTHONPATH=/api
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

