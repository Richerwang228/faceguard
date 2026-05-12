# 🚀 部署指南

## 方式一：Streamlit Cloud（推荐，免费，最简单）

1. 将代码推送到 GitHub
2. 访问 [share.streamlit.io](https://share.streamlit.io)
3. 使用 GitHub 账号登录
4. 选择仓库 `privacy-video-checker`
5. 点击 Deploy，等待 2-3 分钟
6. 获得永久免费 URL（如 `https://yourname-faceguard.streamlit.app`）

## 方式二：Render.com

1. 将代码推送到 GitHub
2. 访问 [render.com](https://render.com)
3. 注册/登录后点击 "New +" → "Web Service"
4. 连接 GitHub 仓库
5. 选择 Free Plan，环境选 Python 3
6. Build Command: `pip install -r requirements.txt`
7. Start Command: `streamlit run app.py --server.port=10000 --server.address=0.0.0.0`
8. 点击 Deploy

或直接使用 `app.json` 一键部署（BluePrint）。

## 方式三：Docker（自有服务器）

```bash
docker-compose up -d
```

访问 `http://你的服务器IP:8501`

## 方式四：本地运行（开发/测试）

```bash
pip install -r requirements.txt
streamlit run app.py
```

访问 `http://localhost:8501`

## 临时演示 URL（当前会话有效）

> **https://f695985d795a77e9-161-81-224-255.serveousercontent.com**
>
> 注意：此为临时隧道，会话结束后会失效。如需永久 URL，请使用上述部署方式。
