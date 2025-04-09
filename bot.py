import nonebot
import sys
import numpy as np
from nonebot import get_driver, on_message
from nonebot.adapters.onebot.v11 import (
    Adapter,
    Bot,
    MessageSegment,
    GroupMessageEvent,
)
from utils.image_processing import ImageProcessor
from utils.logger import logger
from services.queue_manager import message_queue
import asyncio
from typing import Optional

# 全局变量
processor: Optional[ImageProcessor] = None

SYSTEM_FACE_BLACKLIST = {}
EMOJI_BLACKLIST = {"😅"}
TEXT_BLACKLIST = {}

class MessageTask:
    def __init__(self, processor: ImageProcessor, bot: Bot, event: GroupMessageEvent):
        self.processor = processor
        self.bot = bot
        self.event = event

    async def process(self):
        """处理消息主逻辑"""
        try:
            # 检查文本消息
            if await self._check_text_blacklist():
                await self._delete_message()
                return
                
            # 检查系统表情
            if await self._check_face_blacklist():
                await self._delete_message()
                return
                
            # 检查emoji表情
            if await self._check_emoji_blacklist():
                await self._delete_message()
                return
                
            # 原有图片处理逻辑
            for msg in self.event.message:
                if msg.type == "image":
                    url = msg.data["url"]
                    prob = await self.processor.predict_image(url)
                    print(prob)
                    if prob and prob > 0.95:
                        await self._delete_message()
                        await self.bot.call_api(
                            "send_group_msg",
                            group_id=self.event.group_id,
                            message="小猫猫评分" + str(round(prob, 2))
                        )
        except Exception as e:
            logger.error(f"消息处理失败: {str(e)}", exc_info=True)

    async def _check_text_blacklist(self) -> bool:
        """检查文本是否在黑名单中"""
        text = self.event.get_plaintext()
        return any(banned_text in text for banned_text in TEXT_BLACKLIST)

    async def _check_face_blacklist(self) -> bool:
        """检查系统表情是否在黑名单中"""
        for msg in self.event.message:
            if msg.type == "face":
                face_id = int(msg.data["id"])
                if face_id in SYTEM_FACE_BLACKLIST:
                    return True
        return False

    async def _check_emoji_blacklist(self) -> bool:
        """检查emoji是否在黑名单中"""
        text = str(self.event.message)
        return any(emoji in text for emoji in EMOJI_BLACKLIST)

    async def _delete_message(self):
        """撤回消息"""
        try:
            await self.bot.delete_msg(message_id=self.event.message_id)
            logger.info(f"已撤回消息 {self.event.message_id}")
            
        except Exception as e:
            logger.error(f"撤回失败: {str(e)}")

# 初始化NoneBot
nonebot.init()
driver = get_driver()
driver.register_adapter(Adapter)

# 创建消息处理器
matcher = on_message(priority=10)

@matcher.handle()
async def handle_message(bot: Bot, event: GroupMessageEvent):
    """消息处理入口"""
    global processor
    try:
        if processor is None:
            logger.error("Processor未初始化!")
            return

        task = MessageTask(processor=processor, bot=bot, event=event)
        await message_queue.put(task)
    except Exception as e:
        logger.error(f"消息入队失败: {str(e)}")

async def init_processor():
    """异步初始化处理器"""
    global processor
    processor = ImageProcessor("models/v1.h5")
    # 预热模型
    await processor.initialize()
    dummy_data = np.zeros((224, 224, 3), dtype=np.float32)
    _ = processor.model.predict(dummy_data[np.newaxis, ...])
    logger.info("模型预热完成")

@driver.on_startup
async def startup():
    """启动时初始化"""
    try:
        await init_processor()
        await message_queue.start()
        logger.info("服务启动完成")
    except Exception as e:
        logger.critical(f"启动失败: {str(e)}")
        raise

@driver.on_shutdown
async def shutdown():
    """关闭时清理"""
    if processor:
        await processor.close()
    logger.info("服务已关闭")

if __name__ == "__main__":
    # 解决Windows上的事件循环问题
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    nonebot.run()

