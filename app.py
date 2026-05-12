import streamlit as st
import os
import tempfile
import shutil
import sys
import base64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.video_processor import VideoProcessor
from core.report_generator import ReportGenerator

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="FaceGuard — 视频打码自检",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# GLOBAL CSS — Dark Security Dashboard Theme
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #1a2332;
    --bg-hover: #232d3f;
    --border-subtle: #2a3441;
    --border-glow: #3b4d63;
    --text-primary: #e8ecf1;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-green: #22c55e;
    --accent-green-glow: rgba(34, 197, 94, 0.15);
    --accent-red: #ef4444;
    --accent-red-glow: rgba(239, 68, 68, 0.15);
    --accent-amber: #f59e0b;
    --accent-amber-glow: rgba(245, 158, 11, 0.15);
    --accent-blue: #3b82f6;
    --accent-blue-glow: rgba(59, 130, 246, 0.15);
    --font-body: 'Noto Sans SC', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* Global overrides */
.stApp {
    background: var(--bg-primary) !important;
    font-family: var(--font-body) !important;
}

/* Hide Streamlit header/menu */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    font-family: var(--font-body) !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(37, 99, 235, 0.5) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
.stButton > button:disabled {
    background: #374151 !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}

/* Secondary button */
button[kind="secondary"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
}

/* Upload area */
.stFileUploader > div > div {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-glow) !important;
    border-radius: 16px !important;
    padding: 40px 20px !important;
    transition: all 0.3s ease !important;
}
.stFileUploader > div > div:hover {
    border-color: var(--accent-blue) !important;
    background: var(--bg-hover) !important;
}
.stFileUploader small {
    color: var(--text-muted) !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
    border-radius: 4px !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-size: 14px !important;
}
.streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* Sliders */
st-slider {
    color: var(--accent-blue) !important;
}

/* Spinner */
.stSpinner > div {
    border-color: var(--accent-blue) !important;
    border-top-color: transparent !important;
}

/* Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(37, 99, 235, 0.2); }
    50% { box-shadow: 0 0 40px rgba(37, 99, 235, 0.4); }
}
@keyframes scanline {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100%); }
}

.animate-fade-in {
    animation: fadeInUp 0.5s ease forwards;
}
.animate-pulse-glow {
    animation: pulse-glow 2s ease-in-out infinite;
}

/* Custom metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: var(--border-glow);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 32px;
    font-weight: 900;
    font-family: var(--font-mono);
    margin: 8px 0;
}
.metric-label {
    font-size: 13px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Status banners */
.status-pass {
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05));
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.status-fail {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.status-warn {
    background: linear-gradient(135deg, rgba(245,158,11,0.1), rgba(245,158,11,0.05));
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

/* Person cards */
.person-card {
    background: var(--bg-card);
    border-left: 4px solid;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 8px 0;
}
.person-card.protected { border-color: var(--accent-green); }
.person-card.risk { border-color: var(--accent-amber); }
.person-card.unprotected { border-color: var(--accent-red); }

/* Screenshot gallery */
.screenshot-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
}
.screenshot-item {
    border-radius: 12px;
    overflow: hidden;
    border: 2px dashed var(--accent-red);
    transition: transform 0.3s ease;
}
.screenshot-item:hover {
    transform: scale(1.02);
}
.screenshot-item img {
    width: 100%;
    display: block;
}

/* Timeline bar */
.timeline-container {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
}
.timeline-track {
    height: 24px;
    border-radius: 6px;
    display: flex;
    overflow: hidden;
    background: var(--bg-secondary);
}
.timeline-segment {
    height: 100%;
    transition: all 0.3s ease;
}
.timeline-segment:hover {
    filter: brightness(1.3);
}

/* Hero section */
.hero-title {
    font-size: 48px;
    font-weight: 900;
    background: linear-gradient(135deg, #e8ecf1, #3b82f6, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 18px;
    color: var(--text-secondary);
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: var(--accent-green);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    margin-top: 16px;
}

/* Scan line effect */
.scan-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
    opacity: 0.3;
    pointer-events: none;
    z-index: 9999;
}

/* Footer */
.app-footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 12px;
    padding: 40px 0 20px;
    border-top: 1px solid var(--border-subtle);
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# STATE MANAGEMENT
# ============================================================================
if "results" not in st.session_state:
    st.session_state.results = None
if "processing" not in st.session_state:
    st.session_state.processing = False


# ============================================================================
# HERO SECTION
# ============================================================================
st.markdown("""
<div style="text-align: center; padding: 40px 0 30px;">
    <div class="hero-title">🔒 FaceGuard</div>
    <div class="hero-subtitle">视频打码隐私自检工具 — 确保每一帧都安全</div>
    <div class="hero-badge">✓ 本地处理 · 零上传 · 开源可审计</div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ============================================================================
# MAIN INTERFACE
# ============================================================================
main_col, side_col = st.columns([2, 1])

with main_col:
    # Upload area
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span style="font-size: 20px; font-weight: 700; color: var(--text-primary);">📤 上传视频</span>
        <span style="font-size: 13px; color: var(--text-muted); margin-left: 8px;">支持 MP4, MOV, MKV, WebM · 最大 500MB</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["mp4", "mov", "mkv", "webm"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # File size check
        MAX_SIZE = 500 * 1024 * 1024
        if uploaded_file.size > MAX_SIZE:
            st.error(f"❌ 文件过大 ({uploaded_file.size / (1024*1024):.0f}MB)，最大支持 500MB")
        else:
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                video_path = tmp.name

            # Show video preview
            st.video(video_path)

            # Detect button + Advanced settings
            btn_col, adv_col = st.columns([1, 1])
            with btn_col:
                detect_clicked = st.button(
                    "🔍 开始检测",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.processing,
                )

            with adv_col:
                with st.expander("⚙️ 高级设置"):
                    st.markdown("<span style='color: var(--text-muted); font-size: 12px;'>如无特殊需求，保持默认即可</span>", unsafe_allow_html=True)
                    sample_rate = st.slider("抽帧率 (fps)", 1, 10, 5,
                                           help="越高越精确，但处理时间越长")
                    min_face_size = st.slider("最小人脸尺寸", 32, 128, 48,
                                             help="小于此值的人脸将被忽略")
                    clarity_threshold = st.slider("检测严格度", 30, 80, 50,
                                                 help="越低越严格，越容易发现问题")

            # Processing
            if detect_clicked and not st.session_state.processing:
                st.session_state.processing = True
                st.session_state.results = None

                progress_placeholder = st.empty()
                status_placeholder = st.empty()

                with progress_placeholder.container():
                    progress_bar = st.progress(0)

                def on_progress(pct, current, total, ts):
                    progress_bar.progress(min(1.0, pct), text=f"🔄 检测中... {current}/{total} 帧 · {ts}")

                try:
                    processor = VideoProcessor(
                        sample_rate=int(sample_rate),
                        min_face_size=int(min_face_size),
                        blur_threshold=float(clarity_threshold),
                        adaptive_sampling=True,
                    )

                    results = processor.process(video_path, progress_callback=on_progress)

                    # Generate screenshots
                    screenshots_dir = tempfile.mkdtemp()
                    screenshots = processor.extract_screenshots(
                        video_path, results['problem_frames'],
                        screenshots_dir, max_screenshots=20,
                    )

                    # Generate report
                    reporter = ReportGenerator()
                    st.session_state.results = {
                        **results,
                        "screenshots": screenshots,
                        "text_report": reporter.generate_text_report(results),
                        "html_report": reporter.generate_html_report(results, screenshots),
                    }

                    progress_placeholder.empty()
                    status_placeholder.success("✅ 检测完成！")

                except Exception as e:
                    progress_placeholder.empty()
                    status_placeholder.error(f"❌ 处理失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

                finally:
                    st.session_state.processing = False
                    # Cleanup
                    for p in [video_path]:
                        try:
                            if p and os.path.exists(p):
                                os.unlink(p)
                        except Exception:
                            pass
                    if screenshots_dir:
                        try:
                            shutil.rmtree(screenshots_dir, ignore_errors=True)
                        except Exception:
                            pass

                st.rerun()


# ============================================================================
# RESULTS DISPLAY
# ============================================================================
if st.session_state.results:
    results = st.session_state.results
    s = results['summary']
    v = results['video_info']

    st.divider()

    # Status Banner
    st.markdown("""
    <div style="margin: 20px 0 30px;">
        <span style="font-size: 20px; font-weight: 700; color: var(--text-primary);">📊 检测结果</span>
    </div>
    """, unsafe_allow_html=True)

    if s['status'] == 'passed':
        st.markdown(f"""
        <div class="status-pass animate-fade-in">
            <div style="font-size: 48px; margin-bottom: 8px;">✅</div>
            <div style="font-size: 24px; font-weight: 700; color: #22c55e;">检测通过</div>
            <div style="color: #94a3b8; margin-top: 8px;">
                未发现明显问题 · {s['total_persons']} 个人物 · {v['sampled_frames']} 帧已检测
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        risk_color = {"low": "#f59e0b", "medium": "#ef4444", "high": "#ef4444", "critical": "#ef4444"}.get(s['risk_level'], '#ef4444')
        st.markdown(f"""
        <div class="status-fail animate-fade-in">
            <div style="font-size: 48px; margin-bottom: 8px;">⚠️</div>
            <div style="font-size: 24px; font-weight: 700; color: {risk_color};">发现问题</div>
            <div style="color: #94a3b8; margin-top: 8px;">
                {s['problem_frames_count']} 个问题帧 ({s['problem_frames_ratio']*100:.1f}%) · {s['total_persons']} 个人物 · 风险等级: {s['risk_level'].upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Metrics Row
    metrics = st.columns(4)
    metric_data = [
        ("⏱️ 时长", v['duration_str']),
        ("🎞️ 检测帧数", f"{v['sampled_frames']} 帧"),
        ("👥 人物数", f"{s['total_persons']} 人"),
        ("⚠️ 问题帧", f"{s['problem_frames_count']} 帧"),
    ]
    for col, (label, value) in zip(metrics, metric_data):
        with col:
            st.markdown(f"""
            <div class="metric-card animate-fade-in">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color: var(--text-primary);">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # Person Status Cards
    st.markdown("""
    <div style="margin: 30px 0 16px;">
        <span style="font-size: 18px; font-weight: 700; color: var(--text-primary);">👤 人物状态详情</span>
    </div>
    """, unsafe_allow_html=True)

    for p in results['persons']:
        icon = {"protected": "✅", "partial_risk": "⚠️", "unprotected": "❌"}.get(p['status'], "❓")
        label = {"protected": "已保护", "partial_risk": "漏打风险", "unprotected": "未打码"}.get(p['status'], p['status'])
        css_class = {"protected": "protected", "partial_risk": "risk", "unprotected": "unprotected"}.get(p['status'], "")
        color = {"protected": "#22c55e", "partial_risk": "#f59e0b", "unprotected": "#ef4444"}.get(p['status'], "#94a3b8")
        problem_count = len(p['problem_timestamps'])

        st.markdown(f"""
        <div class="person-card {css_class} animate-fade-in">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 18px; font-weight: 700; color: {color};">{icon} 人物 #{p['track_id']}</span>
                    <span style="font-size: 14px; color: var(--text-secondary); margin-left: 12px;">{label}</span>
                </div>
                <div style="text-align: right;">
                    <span style="font-family: var(--font-mono); font-size: 16px; font-weight: 700; color: var(--text-primary);">{p['frames_protected']}/{p['frames_total']}</span>
                    <span style="font-size: 12px; color: var(--text-muted);"> 帧已打码</span>
                </div>
            </div>
            <div style="margin-top: 8px; font-size: 13px; color: var(--text-muted);">
                首次出现: {p['first_appearance']} · 最后出现: {p['last_appearance']}
                {f' · <span style="color: {color};">{problem_count} 处问题</span>' if problem_count else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Screenshots Gallery
    screenshots = results.get('screenshots', [])
    if screenshots:
        st.markdown("""
        <div style="margin: 30px 0 16px;">
            <span style="font-size: 18px; font-weight: 700; color: var(--text-primary);">📸 问题帧截图</span>
            <span style="font-size: 13px; color: var(--text-muted); margin-left: 8px;">红色虚线框标注了检测到的人脸位置</span>
        </div>
        """, unsafe_allow_html=True)

        # Display in grid using Streamlit columns
        cols_per_row = 3
        for i in range(0, min(len(screenshots), 9), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(screenshots) and os.path.exists(screenshots[idx]):
                    with col:
                        st.image(screenshots[idx], use_container_width=True)

    # Timeline (simplified)
    if results['problem_frames']:
        st.markdown("""
        <div style="margin: 30px 0 16px;">
            <span style="font-size: 18px; font-weight: 700; color: var(--text-primary);">⏱️ 问题时间轴</span>
        </div>
        """, unsafe_allow_html=True)

        # Build timeline HTML
        timeline_html = '<div class="timeline-container">'
        for pf in results['problem_frames'][:15]:
            timeline_html += f"""
            <div style="display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border-subtle);">
                <span style="font-family: var(--font-mono); font-size: 14px; color: var(--accent-red); min-width: 100px;">⏱ {pf['timestamp_str']}</span>
                <span style="color: var(--text-secondary); font-size: 14px;">人物 #{pf['track_id']} — 人脸露出</span>
            </div>
            """
        if len(results['problem_frames']) > 15:
            timeline_html += f'<div style="text-align: center; padding: 12px; color: var(--text-muted); font-size: 13px;">... 还有 {len(results["problem_frames"]) - 15} 个问题帧</div>'
        timeline_html += '</div>'
        st.markdown(timeline_html, unsafe_allow_html=True)

    # Action buttons
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        st.download_button(
            label="📥 下载 HTML 报告",
            data=results['html_report'],
            file_name="FaceGuard_Report.html",
            mime="text/html",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            label="📄 下载文字报告",
            data=results['text_report'],
            file_name="FaceGuard_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with dl_col3:
        if st.button("🔄 检测新视频", use_container_width=True):
            st.session_state.results = None
            st.rerun()


# ============================================================================
# SIDEBAR — Only show when no results
# ============================================================================
if not st.session_state.results:
    with side_col:
        st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 16px; padding: 24px;">
            <div style="font-size: 16px; font-weight: 700; color: var(--text-primary); margin-bottom: 16px;">💡 如何使用</div>
            <div style="font-size: 14px; color: var(--text-secondary); line-height: 1.8;">
                <div style="margin-bottom: 12px;"><span style="color: var(--accent-blue); font-weight: 700;">1.</span> 上传已打码的视频文件</div>
                <div style="margin-bottom: 12px;"><span style="color: var(--accent-blue); font-weight: 700;">2.</span> 点击「开始检测」按钮</div>
                <div style="margin-bottom: 12px;"><span style="color: var(--accent-blue); font-weight: 700;">3.</span> 等待 AI 分析完成</div>
                <div><span style="color: var(--accent-blue); font-weight: 700;">4.</span> 查看问题帧和报告</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 16px; padding: 24px; margin-top: 16px;">
            <div style="font-size: 16px; font-weight: 700; color: var(--text-primary); margin-bottom: 16px;">🔒 隐私承诺</div>
            <div style="font-size: 14px; color: var(--text-secondary); line-height: 1.8;">
                <div style="margin-bottom: 8px;">✓ 视频在本地处理，不上传服务器</div>
                <div style="margin-bottom: 8px;">✓ 处理完成后立即删除</div>
                <div style="margin-bottom: 8px;">✓ 不保存任何帧截图</div>
                <div>✓ 开源可审计，代码透明</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 16px; padding: 24px; margin-top: 16px;">
            <div style="font-size: 16px; font-weight: 700; color: var(--text-primary); margin-bottom: 16px;">🛡️ 支持的打码类型</div>
            <div style="font-size: 14px; color: var(--text-secondary); line-height: 1.8;">
                <div style="margin-bottom: 6px;">✓ 高斯模糊</div>
                <div style="margin-bottom: 6px;">✓ 马赛克 / 像素块</div>
                <div style="margin-bottom: 6px;">✓ 黑色 / 白色方块</div>
                <div>✓ 贴纸 / 表情包遮挡</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="app-footer">
    FaceGuard · 视频打码隐私自检工具 · 本地处理 · 开源可审计<br>
    <span style="font-size: 11px; color: #475569;">Made with 🔒 for content creators</span>
</div>
""", unsafe_allow_html=True)
