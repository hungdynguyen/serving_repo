# Sử dụng base image Python
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
# Các biến môi trường khác cho API có thể được đặt ở đây hoặc trong docker-compose

WORKDIR /app

# Sao chép các file requirements
# Giả sử requirements_common.txt và requirements_api.txt nằm trong cùng thư mục build context
COPY requirements_api.txt .

# Cài đặt Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Sao chép code API và các file cần thiết
COPY src/ . 
# COPY data/ ./data/ # Nếu API của bạn cần thư mục data này bên trong image

EXPOSE 8000 

# Lệnh để chạy Uvicorn server
# Giả sử trong main.py, instance FastAPI của bạn tên là 'app'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]