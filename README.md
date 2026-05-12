# 🔒 视频打码隐私检测工具

**Privacy Video Checker — AI 驱动的视频打码合规检测工具**

上传已打码的视频，AI 自动检测是否有人脸/敏感信息露出，生成可视化检测报告。

[English](#english) | [中文](#中文)

---

## 中文

### 功能特性

- **人脸检测**：支持正面 + 侧面人脸，高速运动场景优化
- **多目标跟踪**：运动预测算法，人物快速移动不丢失
- **自适应抽帧**：根据视频时长智能调整采样率
- **四种打码检测**：模糊 / 马赛克 / 黑块 / 贴纸
- **问题帧定位**：精确到毫秒的时间戳，附带标注截图
- **可视化报告**：HTML 报告 + 时间轴 + 截图画廊
- **隐私优先**：纯本地处理，零上传，处理完立即删除
- **傻瓜式 UI**：一键检测，开箱即用

### 截图

> 运行 `streamlit run app.py` 后，在浏览器中打开即可看到界面。
>
> 截图将在首次运行后自动展示。

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/privacy-video-checker.git
cd privacy-video-checker

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 使用

```bash
streamlit run app.py
```

然后在浏览器中打开 `http://localhost:8501`，上传视频即可开始检测。

### Docker 部署

```bash
docker-compose up -d
```

访问 `http://localhost:8501`

### 技术原理

1. **人脸检测**：OpenCV Haar Cascade 级联分类器，支持正面 + 侧面人脸检测，NMS 去重
2. **目标跟踪**：基于 IoU 的跟踪器，带运动预测（指数平滑速度估计），支持高速移动场景
3. **自适应抽帧**：根据视频时长智能调整（≤5s 用 10fps，长视频自动降低），平衡精度与速度
4. **打码判定**：综合清晰度评分（拉普拉斯方差 + Sobel 梯度 + 局部对比度 + 块方差），检测模糊/马赛克/黑块/贴纸四种打码方式
5. **报告生成**：汇总每个人的保护状态，标记问题帧并生成带标注的截图

### 隐私承诺

- 视频处理完全在本地进行
- 不调用任何云端 API
- 处理完成后临时文件立即删除
- 开源可审计，代码透明

---

## English

### Features

- Face detection: frontal + profile, optimized for fast motion
- Multi-target tracking: motion prediction for high-speed scenes
- Adaptive sampling: intelligently adjusts frame rate by video length
- 4 mosaic types: blur / pixelation / black block / sticker
- Problem frame localization: millisecond-precision timestamps with annotated screenshots
- Visual reports: HTML report + timeline + screenshot gallery
- Privacy-first: pure local processing, zero upload, auto-deletion
- Foolproof UI: one-click detection, works out of the box

### Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Docker

```bash
docker-compose up -d
```

### Tech Stack

- Python 3.10+
- OpenCV (DNN face detection)
- Streamlit (Web UI)
- NumPy / Pillow

---

## License

[MIT License](./LICENSE)
