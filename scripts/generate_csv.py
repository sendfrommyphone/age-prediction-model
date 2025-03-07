import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 配置参数
IMAGE_DIR = "./UTKface"  # 图片存放目录
SAVE_PATH = "./"                   # CSV保存路径
SPLIT_RATIO = 0.2                  # 验证集比例
RANDOM_SEED = 42                   # 随机种子

def parse_filename(filename):
    """解析文件名，提取元数据"""
    try:
        # 去除所有.jpg后缀，提取主文件名
        base = filename.split(".jpg")[0]
        parts = base.split("_")
        
        # 解析字段（格式：年龄_性别_种族_时间戳）
        age = int(parts[0])      # 第1个字段：年龄
        gender = int(parts[1])   # 第2个字段：性别
        race = int(parts[2])     # 第3个字段：种族
        timestamp = parts[3]    # 第4个字段：时间戳
        
        return age, gender, race, timestamp
    except Exception as e:
        print(f"解析失败: {filename} - Error: {str(e)}")
        return None

# 收集数据
data = []
for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    
    parsed = parse_filename(filename)
    if parsed:
        age, gender, race, timestamp = parsed
        
        # 数据有效性检查
        if 1 <= age <= 100 and gender in [0, 1] and race >= 0:  # 根据实际情况修改约束
            data.append({
                "filepath": os.path.abspath(os.path.join(IMAGE_DIR, filename)),
                "age": age,
                "gender": gender,
                "race": race,
                "timestamp": timestamp
            })

# 创建DataFrame
df = pd.DataFrame(data)

# 分层分割数据集（保持年龄分布）
train_df, val_df = train_test_split(
    df,
    test_size=SPLIT_RATIO,
    random_state=RANDOM_SEED,
    stratify=df["age"]  # 若数据量小可移除stratify
)

# 保存CSV
train_csv = os.path.join(SAVE_PATH, "train.csv")
val_csv = os.path.join(SAVE_PATH, "val.csv")

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)

# 打印统计信息
print(f"成功解析 {len(df)} 张图片")
print(f"训练集: {len(train_df)} 张 | 验证集: {len(val_df)} 张")
print(f"年龄分布示例:\n{df['age'].describe()}")
print(f"性别分布:\n{df['gender'].value_counts()}")
print(f"种族分布:\n{df['race'].value_counts()}")