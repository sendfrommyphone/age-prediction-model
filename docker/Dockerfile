# 基础镜像（推荐使用轻量级版本）
FROM python:3.9-slim

# 设置工作目录（容器内路径）
WORKDIR /app

# 先复制依赖文件（利用Docker缓存优化构建速度）
COPY requirements.txt .

# 安装依赖（使用国内镜像加速）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制应用代码和模型文件
COPY app.py .
COPY model.onnx .

# 暴露端口（与代码中uvicorn的端口一致）
EXPOSE 8080

# 启动命令（生产环境建议去掉--reload）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]