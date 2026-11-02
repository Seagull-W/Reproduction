## 下载权重
```bash
pip install -U huggingface_hub

bash download.sh /data/models/MAI-Gym-9B
```
## 推理
推理后端选择vllm，在两个终端中分别启动服务，加载两份模型副本

```bash
source .venv/bin/activate
bash serve.sh 0 8000 /data/models/MAI-Gym-9B
```
```bash
source .venv/bin/activate
bash serve.sh 1 8001 /data/models/MAI-Gym-9B
```

## agent.py

环境截图（PNG 字节）+ 任务与历史消息
                  ↓
          agent.py 发送 HTTP 请求
                  ↓
     GPU 0:8000 或 GPU 1:8001
                  ↓
      完整响应：回答、token 用量、停止原因

