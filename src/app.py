from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
import onnxruntime as ort  # 使用 ONNX Runtime

app = FastAPI()

# 加载 ONNX 模型
model = ort.InferenceSession("model.onnx")  # 确保 model.onnx 文件路径正确

# 获取模型输入名称（例如 "input"）
input_name = model.get_inputs()[0].name

# 预处理函数（输出需为 NumPy 数组，而非 PyTorch 张量）
def preprocess(image_bytes: bytes) -> np.ndarray:
    # 转换为 PIL 图像
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # 调整尺寸
    image = image.resize((224, 224))
    # 转换为 NumPy 数组并调整维度顺序 (HWC → CHW)
    image = np.array(image).transpose(2, 0, 1).astype(np.float32)
    # 归一化（假设模型需要 [0,1] 范围）
    image = image / 255.0
    # 添加 batch 维度 (shape 变为 [1, 3, 224, 224])
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 读取上传的文件
        image_bytes = await file.read()
        # 预处理
        input_data = preprocess(image_bytes)
        # ONNX 推理
        ort_inputs = {input_name: input_data}
        ort_outputs = model.run(None, ort_inputs)  # 输出是一个列表，例如 [output]
        # 后处理（假设输出是分类 logits）
        prediction = np.argmax(ort_outputs[0], axis=1)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)