from config.blacklist import *
from utils.logger import logger
import emoji

def is_blacklist_content(message: str) -> bool:
    """检查消息是否包含黑名单内容"""
    try:
        # 标准化处理
        demojized = emoji.demojize(message)
        
        # 检查系统表情
        if "[CQ:face" in message:
            ids = [int(s.split("=")[1]) for s in message.split(",") if "id=" in s]
            if any(id in SYSTEM_FACE_BLACKLIST for id in ids):
                return True
        
        # 检查Emoji
        if any(e in demojized for e in EMOJI_BLACKLIST):
            return True
        
        # 检查关键词
        if any(kw in message for kw in TEXT_BLACKLIST):
            return True
        
        return False
    except Exception as e:
        logger.error(f"内容过滤失败: {str(e)}")
        return False