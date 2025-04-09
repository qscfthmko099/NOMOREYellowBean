# 修复脚本 fix_model.py
import tensorflow as tf
import numpy as np

try:
    # 尝试用低级API加载
    model = tf.keras.models.load_model("models/v1.h5", compile=False)
    model.save("models/v1_fixed.h5")  # 重新保存为修复版
    print("模型修复成功，请使用 v1_fixed.h5")
except Exception as e:
    print("无法修复原模型，正在创建应急模型...")
    # 创建极简模型替代
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224,224,3)),
        tf.keras.layers.Lambda(lambda x: x, name="bypass")
    ])
    model.save("models/emergency.h5")
    print("已创建应急模型 emergency.h5")