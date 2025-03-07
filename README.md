# 年龄预测回归模型 🎯

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue)](docker/Dockerfile)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-red)](https://pytorch.org/)

基于深度学习的人脸年龄预测系统，集成ConvNeXt与ResNet模型，提供高精度预测和便捷的API服务。

![Prediction Demo](https://via.placeholder.com/800x400.png/007bff/FFFFFF?text=Age+Prediction+Demo)

## 功能特性 ✨
- ​**高精度模型**：最佳模型MAE达到5.25岁（UTKFace测试集）
- ​**高效推理**：支持ONNX Runtime加速，单次预测<150ms
- ​**便捷部署**：提供Docker容器化解决方案
- ​**灵活训练**：支持多种预训练模型微调
- ​**可视化分析**：内置训练过程监控与误差分析工具

## 快速开始 🚀

### 环境要求
- Python 3.8+
- CUDA 11.7 (GPU模式)
- Docker 20.10+ (可选)

### 安装依赖
```bash
git clone https://github.com/yourusername/age-prediction.git
cd age-prediction
pip install -r requirements.txt
