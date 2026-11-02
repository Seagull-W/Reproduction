import requests
import re
import base64

ENDPOINTS = (
    "http://127.0.0.1:8000/v1",
    "http://127.0.0.1:8001/v1",
)


def make_observation_message(text:str,screenshot_png:bytes)->dict:
    """把文字和一张 PNG 截图组成一条 user 消息。
    用data_url:base64封装图片
    """
    encoded=base64.b64encode(screenshot_png).decode("ascii")

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded}",
                },
            },
        ],
    }

def request_model(
    sessions:requests.Session,
    endpoint:str,
    messages:list[dict],
    max_tokens:int,
)->dict:
    """通过request向模型通信，发送指定上下文，接收完整的响应"""
    payload={
        "model":'mai-gym',
        "messages":messages,
        "temperature":0,
        "max_tokens":max_tokens,
    }

    response=sessions.post(
        f"{endpoint}/chat/completions",
        json=payload,
        timeout=(10,600),
    )
    """raise_for_status在报错时抛出异常"""
    response.raise_for_status()

    return response.json()

"""解析模型回答为OSWorld需要的输入：
模型回答：
```python
pyautogui.click(640, 360)
```

解析结果：
pyautogui.click(640, 360)

"""

def parse_action(raw_output:str)->str:
    """提取代码块，或WAIT / DONE / FAIL 特殊动作"""
    text=raw_output.strip()

    if text in {"WAIT","DONE","FAIL"}:
        return text

    blocks = re.findall(
        r"```(?:python[ \t]*\n)?(.*?)```",
        text,
        flags=re.DOTALL,
    )

    if len(blocks)!=1:
        raise ValueError(f"预期一个action，实际得到{len(blocks)}个action")

    action = blocks[0].strip()
    if not action:
        raise ValueError("action为空")

    return action