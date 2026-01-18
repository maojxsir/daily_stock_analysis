# -*- coding: utf-8 -*-
"""
===================================
飞书机器人事件订阅服务
===================================

功能：
1. 接收飞书群聊中@机器人的消息
2. 解析消息中的股票代码或名称
3. 调用股票分析功能
4. 将分析结果推送回飞书群聊

参考文档：
- https://open.feishu.cn/document/ukTMukTMukTM/uYjL24iN2EjL2YTN
- https://open.feishu.cn/document/ukTMukTMukTM/uUTNz4SN1MjL1UzM
"""

import json
import logging
import re
import hashlib
import hmac
import base64
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

import requests
from flask import Flask, request, jsonify

from config import get_config
from main import StockAnalysisPipeline
from analyzer import STOCK_NAME_MAP

logger = logging.getLogger(__name__)

app = Flask(__name__)


class FeishuBotService:
    """
    飞书机器人服务

    处理飞书事件订阅，接收@消息并分析股票
    """

    def __init__(self):
        self.config = get_config()
        self.app_id = self.config.feishu_app_id
        self.app_secret = self.config.feishu_app_secret
        self.encrypt_key = getattr(self.config, 'feishu_encrypt_key', None)

        # 初始化分析管道
        self.pipeline = StockAnalysisPipeline()

        # 飞书 API 基础 URL
        self.api_base = "https://open.feishu.cn/open-apis"

        # 访问令牌缓存
        self._access_token = None
        self._token_expires_at = 0

    def is_configured(self) -> bool:
        """检查配置是否完整"""
        return bool(self.app_id and self.app_secret)

    def get_access_token(self) -> Optional[str]:
        """
        获取飞书访问令牌（tenant_access_token）

        参考：https://open.feishu.cn/document/ukTMukTMukTM/ukDNz4SO0MjL5QzM
        """
        # 检查缓存
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token

        try:
            url = f"{self.api_base}/auth/v3/tenant_access_token/internal"
            payload = {
                "app_id": self.app_id,
                "app_secret": self.app_secret
            }

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get('code') == 0:
                self._access_token = data.get('tenant_access_token')
                # 令牌有效期通常是 2 小时，提前 5 分钟刷新
                expire_time = data.get('expire', 7200)
                self._token_expires_at = time.time() + expire_time - 300
                logger.info("飞书访问令牌获取成功")
                return self._access_token
            else:
                logger.error(f"获取飞书访问令牌失败: {data}")
                return None

        except Exception as e:
            logger.error(f"获取飞书访问令牌异常: {e}")
            return None

    def verify_event_signature(self, timestamp: str, nonce: str, body: str, signature: str) -> bool:
        """
        验证飞书事件签名

        参考：https://open.feishu.cn/document/ukTMukTMukTM/uYjL24iN2EjL2YTN
        """
        if not self.encrypt_key:
            logger.warning("未配置飞书加密密钥，跳过签名验证")
            return True

        # 构造待签名字符串
        string_to_sign = f"{timestamp}{nonce}{self.encrypt_key}{body}"

        # 计算签名
        signature_bytes = hmac.new(
            self.encrypt_key.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha256
        ).digest()
        expected_signature = base64.b64encode(signature_bytes).decode('utf-8')

        # 验证签名
        return hmac.compare_digest(signature, expected_signature)

    def handle_url_verification(self, challenge: str) -> Dict[str, Any]:
        """
        处理 URL 验证请求（飞书首次配置事件订阅时会发送）

        参考：https://open.feishu.cn/document/ukTMukTMukTM/uYjL24iN2EjL2YTN
        """
        return {
            "challenge": challenge
        }

    def extract_stock_codes(self, text: str) -> List[str]:
        """
        从文本中提取股票代码或名称

        支持格式：
        - 股票代码：600519, 000001, 300750
        - 股票名称：贵州茅台、平安银行
        - 混合：600519 或 贵州茅台

        Args:
            text: 用户输入的文本

        Returns:
            股票代码列表
        """
        codes = []

        # 去除@机器人的部分和多余空格
        text = re.sub(r'@[^\s]+', '', text).strip()

        # 1. 提取股票代码（6位数字）
        code_pattern = r'\b([0-9]{6})\b'
        found_codes = re.findall(code_pattern, text)
        codes.extend(found_codes)

        # 2. 从股票名称映射中查找
        for code, name in STOCK_NAME_MAP.items():
            if name in text:
                if code not in codes:
                    codes.append(code)

        # 3. 尝试从常见股票名称中提取（如果代码未找到）
        if not codes:
            # 常见股票名称关键词
            stock_keywords = {
                '茅台': '600519',
                '平安银行': '000001',
                '宁德时代': '300750',
                '比亚迪': '002594',
                '招商银行': '600036',
                '中国平安': '601318',
                '五粮液': '000858',
            }

            for keyword, code in stock_keywords.items():
                if keyword in text:
                    codes.append(code)
                    break

        # 去重
        return list(set(codes))

    def analyze_stock_and_reply(self, stock_codes: List[str], chat_id: str) -> bool:
        """
        分析股票并发送结果到飞书群聊

        Args:
            stock_codes: 股票代码列表
            chat_id: 飞书群聊 ID

        Returns:
            是否成功
        """
        if not stock_codes:
            self.send_message(chat_id, "❌ 未识别到有效的股票代码或名称，请发送股票代码（如：600519）或股票名称（如：贵州茅台）")
            return False

        try:
            # 发送分析中提示
            self.send_message(chat_id, f"🔍 正在分析 {len(stock_codes)} 只股票，请稍候...")

            # 调用分析管道
            results = self.pipeline.run(
                stock_codes=stock_codes,
                dry_run=False,
                send_notification=False  # 不发送通知，我们手动发送到飞书
            )

            if not results:
                self.send_message(chat_id, "❌ 分析失败，请稍后重试")
                return False

            # 生成分析报告
            from notification import NotificationService
            notifier = NotificationService()
            report = notifier.generate_dashboard_report(results)

            # 发送到飞书
            return self.send_message(chat_id, report)

        except Exception as e:
            logger.exception(f"分析股票失败: {e}")
            self.send_message(chat_id, f"❌ 分析过程出错: {str(e)}")
            return False

    def send_message(self, chat_id: str, content: str) -> bool:
        """
        发送消息到飞书群聊

        参考：https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/create

        Args:
            chat_id: 群聊 ID（chat_id）
            content: 消息内容（Markdown 格式）

        Returns:
            是否发送成功
        """
        access_token = self.get_access_token()
        if not access_token:
            logger.error("无法获取飞书访问令牌")
            return False

        try:
            url = f"{self.api_base}/im/v1/messages"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            # 使用富文本消息格式（支持 Markdown）
            # 参考：https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/create
            # 注意：飞书 API v1 版本，receive_id_type 需要作为查询参数或路径参数
            url_with_params = f"{url}?receive_id_type=chat_id"

            payload = {
                "receive_id": chat_id,
                "msg_type": "interactive",
                "content": json.dumps({
                    "config": {
                        "wide_screen_mode": True
                    },
                    "header": {
                        "title": {
                            "tag": "plain_text",
                            "content": "📊 股票分析报告"
                        }
                    },
                    "elements": [
                        {
                            "tag": "div",
                            "text": {
                                "tag": "lark_md",
                                "content": content
                            }
                        }
                    ]
                })
            }

            response = requests.post(url_with_params, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            if result.get('code') == 0:
                logger.info(f"飞书消息发送成功: {chat_id}")
                return True
            else:
                logger.error(f"飞书消息发送失败: {result}")
                return False

        except Exception as e:
            logger.error(f"发送飞书消息异常: {e}")
            return False

    def handle_message_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理消息接收事件

        参考：https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/events/received

        Args:
            event: 事件数据

        Returns:
            响应数据
        """
        try:
            message = event.get('message', {})
            chat_type = message.get('chat_type')
            chat_id = message.get('chat_id')
            content = message.get('content', '')
            mentions = message.get('mentions', [])

            # 只处理群聊消息
            if chat_type != 'group':
                logger.debug(f"忽略非群聊消息: {chat_type}")
                return {"code": 0}

            # 检查是否@了机器人
            bot_open_id = None  # 可以从配置中获取或通过 API 获取
            is_mentioned = False

            # 解析 content（JSON 字符串）
            try:
                content_obj = json.loads(content) if isinstance(content, str) else content
                text = content_obj.get('text', '')
            except:
                text = str(content)

            # 检查 mentions 中是否包含机器人
            if mentions:
                # 简化处理：如果有 mentions，认为@了机器人
                # 更准确的方式是配置机器人的 open_id 并进行匹配
                is_mentioned = True
                logger.debug(f"检测到@操作，mentions: {mentions}")

            if not is_mentioned:
                logger.debug("消息未@机器人，忽略")
                return {"code": 0}

            # 提取股票代码
            stock_codes = self.extract_stock_codes(text)

            if not stock_codes:
                # 发送帮助信息
                help_text = """📖 **使用说明**

发送股票代码或名称，我会为您分析：

**示例：**
- `600519` - 分析贵州茅台
- `贵州茅台` - 分析贵州茅台
- `600519 000001` - 同时分析多只股票

**支持的格式：**
- 6位股票代码（如：600519）
- 股票名称（如：贵州茅台、平安银行）"""
                self.send_message(chat_id, help_text)
                return {"code": 0}

            # 异步处理分析（避免超时）
            # 注意：实际生产环境建议使用任务队列（如 Celery）
            import threading
            thread = threading.Thread(
                target=self.analyze_stock_and_reply,
                args=(stock_codes, chat_id)
            )
            thread.daemon = True
            thread.start()

            return {"code": 0}

        except Exception as e:
            logger.exception(f"处理消息事件失败: {e}")
            return {"code": 0}  # 即使失败也返回成功，避免飞书重试


# 全局服务实例
bot_service = FeishuBotService()


@app.route('/feishu/event', methods=['POST'])
def feishu_event():
    """
    飞书事件订阅回调接口

    参考：https://open.feishu.cn/document/ukTMukTMukTM/uYjL24iN2EjL2YTN
    """
    try:
        # 获取请求头
        timestamp = request.headers.get('X-Lark-Request-Timestamp', '')
        nonce = request.headers.get('X-Lark-Request-Nonce', '')
        signature = request.headers.get('X-Lark-Signature', '')

        # 获取请求体
        body = request.get_data(as_text=True)
        data = request.get_json()

        if not data:
            logger.warning("收到空请求")
            return jsonify({"code": 0}), 200

        # 验证签名（如果配置了加密密钥）
        if bot_service.encrypt_key:
            if not bot_service.verify_event_signature(timestamp, nonce, body, signature):
                logger.warning("事件签名验证失败")
                return jsonify({"code": 1, "msg": "Invalid signature"}), 403

        # 处理 URL 验证
        if data.get('type') == 'url_verification':
            challenge = data.get('challenge', '')
            return jsonify(bot_service.handle_url_verification(challenge))

        # 处理事件
        header = data.get('header', {})
        event_type = header.get('event_type')

        if event_type == 'im.message.receive_v1':
            event = data.get('event', {})
            result = bot_service.handle_message_event(event)
            return jsonify(result)

        # 其他事件类型暂时忽略
        logger.debug(f"未处理的事件类型: {event_type}")
        return jsonify({"code": 0}), 200

    except Exception as e:
        logger.exception(f"处理飞书事件异常: {e}")
        return jsonify({"code": 0}), 200  # 返回成功避免飞书重试


@app.route('/feishu/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "configured": bot_service.is_configured(),
        "timestamp": datetime.now().isoformat()
    })


def run_bot_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """
    运行飞书机器人服务器

    Args:
        host: 监听地址
        port: 监听端口
        debug: 是否启用调试模式
    """
    if not bot_service.is_configured():
        logger.error("飞书机器人配置不完整，请设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
        return

    logger.info(f"飞书机器人服务启动: http://{host}:{port}")
    logger.info(f"事件订阅回调地址: http://your-domain.com/feishu/event")
    logger.info("请确保该地址可以从公网访问（可使用 ngrok 等工具）")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import sys

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )

    # 从命令行参数获取配置
    host = '0.0.0.0'
    port = 5000
    debug = False

    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    if len(sys.argv) > 2:
        debug = sys.argv[2].lower() == 'true'

    run_bot_server(host=host, port=port, debug=debug)
