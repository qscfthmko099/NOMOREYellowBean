import aiohttp
import asyncio
import numpy as np
from PIL import Image
import io
import hashlib
import logging
from typing import Optional, Tuple
import tensorflow as tf
from pathlib import Path

logger = logging.getLogger("emoji_bot")

class ImageProcessor:
    def __init__(self, model_path: str):
        """
        初始化图片处理器
        :param model_path: 模型文件路径
        """
        self.model_path = Path(model_path)
        self._model = None  # 统一使用下划线前缀的私有变量
        self._session = None
        self._connector = None
        self._lock = asyncio.Lock()
        self._cache = {}
        self._cache_keys = []

    async def initialize(self):
        """初始化模型和网络连接"""
        async with self._lock:
            if self._model is None:
                await self._load_model()
            if self._connector is None:
                self._connector = aiohttp.TCPConnector(
                    ssl=False,
                    force_close=True,
                    limit=4
                )
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    connector=self._connector,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Referer": "https://qun.qq.com/"
                    },
                    timeout=aiohttp.ClientTimeout(total=15)
                )

    async def _load_model(self):
        """加载TensorFlow模型"""
        for i in range(3):
            try:
                self._model = tf.keras.models.load_model(str(self.model_path))
                dummy = np.zeros((1, 640, 640, 3), dtype=np.float32)
                _ = self._model.predict(dummy)
                logger.info(f"模型加载成功 (尝试 {i+1}/3)")
                return
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                if i < 2:
                    await asyncio.sleep(1)
        raise RuntimeError("模型加载失败")

    async def download_image(self, url: str) -> Tuple[Optional[bytes], Optional[str]]:
        """下载图片"""
        try:
            if url.startswith("https://"):
                url = url.replace("https://", "http://", 1)
                
            if self._session is None or self._session.closed:
                await self.initialize()
                
            async with self._session.get(url) as resp:
                resp.raise_for_status()
                if not resp.content_type.startswith('image/'):
                    return None, None
                data = await resp.read()
                return data, hashlib.md5(data).hexdigest()
        except Exception as e:
            logger.error(f"下载失败: {str(e)}")
            return None, None

    @staticmethod
    def preprocess_image(data: bytes) -> Optional[np.ndarray]:
        """图片预处理"""
        try:
            img = Image.open(io.BytesIO(data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return np.array(img.resize((640, 640))) / 255.0
        except Exception as e:
            logger.error(f"图片处理失败: {str(e)}")
            return None

    def _cached_predict(self, img_array: np.ndarray) -> float:
        """带缓存的预测"""
        array_hash = hashlib.md5(img_array.tobytes()).hexdigest()
        if array_hash in self._cache:
            return self._cache[array_hash]
            
        result = float(self._model.predict(img_array[np.newaxis, ...])[0][0])
        self._cache[array_hash] = result
        self._cache_keys.append(array_hash)
        
        if len(self._cache) > 512:
            oldest = self._cache_keys.pop(0)
            del self._cache[oldest]
            
        return result

    async def predict_image(self, url: str) -> Optional[float]:
        """完整预测流程"""
        try:
            data, _ = await self.download_image(url)
            if not data:
                return None
                
            img_array = self.preprocess_image(data)
            if img_array is None:
                return None
                
            return self._cached_predict(img_array)
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            return None

    async def close(self):
        """清理资源"""
        async with self._lock:
            try:
                if self._session:
                    await self._session.close()
                if self._connector:
                    await self._connector.close()
                self._cache.clear()
                self._cache_keys.clear()
            except Exception as e:
                logger.error(f"资源关闭异常: {str(e)}")

    # 兼容性属性访问（可选）
    @property
    def model(self):
        """提供对_model的安全访问（可选）"""
        if self._model is None:
            raise RuntimeError("模型未初始化")
        return self._model