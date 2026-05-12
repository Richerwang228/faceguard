import json
import os
from typing import Dict, List
from datetime import datetime


class ReportGenerator:
    """Generate human-readable reports from processing results."""

    STATUS_ICONS = {
        'passed': '✅',
        'failed': '❌',
        'warning': '⚠️',
        'none': '✅',
        'low': '⚠️',
        'medium': '❌',
        'high': '❌',
        'critical': '❌',
        'protected': '✅',
        'partial_risk': '⚠️',
        'unprotected': '❌',
    }

    STATUS_LABELS = {
        'protected': '已保护',
        'partial_risk': '漏打风险',
        'unprotected': '未打码',
    }

    def generate_text_report(self, results: Dict) -> str:
        """Generate a plain text report for terminal display."""
        v = results['video_info']
        s = results['summary']
        persons = results['persons']
        problems = results['problem_frames']

        lines = []
        lines.append("=" * 60)
        lines.append("     视频打码质量检测报告")
        lines.append("=" * 60)
        lines.append("")

        # Video info
        lines.append("【视频信息】")
        lines.append(f"  文件名: {v['filename']}")
        lines.append(f"  时长: {v['duration_str']}")
        lines.append(f"  分辨率: {v['width']}x{v['height']}")
        lines.append(f"  帧率: {v['fps']:.1f} fps")
        lines.append(f"  检测帧数: {v['sampled_frames']} (抽帧率: {v['sample_rate']}fps)")
        lines.append("")

        # Summary
        icon = self.STATUS_ICONS.get(s['status'], '❓')
        risk = self.STATUS_ICONS.get(s['risk_level'], '❓')
        lines.append("【检测摘要】")
        lines.append(f"  整体状态: {icon} {'通过' if s['status'] == 'passed' else '未通过'}")
        lines.append(f"  风险等级: {risk} {s['risk_level'].upper()}")
        lines.append(f"  检测到人物: {s['total_persons']} 人")
        lines.append(f"  问题帧: {s['problem_frames_count']} / {s['sampled_frames']} ({s['problem_frames_ratio']*100:.2f}%)")
        lines.append("")

        # Person details
        if persons:
            lines.append("【人物状态】")
            for i, p in enumerate(persons, 1):
                icon = self.STATUS_ICONS.get(p['status'], '❓')
                label = self.STATUS_LABELS.get(p['status'], p['status'])
                lines.append(f"  人物 #{p['track_id']}: {icon} {label}")
                lines.append(f"    帧数: {p['frames_protected']}/{p['frames_total']} 已打码")
                if p['problem_timestamps']:
                    lines.append(f"    问题帧数: {len(p['problem_timestamps'])}")
                if p['detected_methods']:
                    methods_str = ', '.join(p['detected_methods'])
                    lines.append(f"    检测到的打码方式: {methods_str}")
            lines.append("")

        # Problem frames
        if problems:
            lines.append("【问题帧列表】")
            for p in problems[:20]:  # Limit to 20 in text
                lines.append(f"  {p['timestamp_str']} | 人物#{p['track_id']} | 人脸露出")
            if len(problems) > 20:
                lines.append(f"  ... 还有 {len(problems) - 20} 个问题帧")
            lines.append("")

        # Recommendations
        lines.append("【建议】")
        recs = self._generate_recommendations(persons)
        for r in recs:
            lines.append(f"  💡 {r}")

        lines.append("")
        lines.append("=" * 60)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return '\n'.join(lines)

    def generate_json_report(self, results: Dict) -> str:
        """Generate a JSON report."""
        report = {
            'version': '1.0.0',
            'generated_at': datetime.now().isoformat(),
            'video_info': results['video_info'],
            'summary': results['summary'],
            'persons': results['persons'],
            'problem_frames': [
                {
                    'timestamp': p['timestamp_str'],
                    'frame_idx': p['frame_idx'],
                    'track_id': p['track_id'],
                    'bbox': p['bbox'],
                }
                for p in results['problem_frames']
            ],
            'recommendations': self._generate_recommendations(results['persons']),
        }
        return json.dumps(report, indent=2, ensure_ascii=False)

    def generate_html_report(self, results: Dict, screenshot_paths: List[str] = None) -> str:
        """Generate an HTML report with embedded screenshots."""
        v = results['video_info']
        s = results['summary']
        persons = results['persons']
        problems = results['problem_frames']

        status_color = {
            'passed': '#22c55e',
            'failed': '#ef4444',
            'warning': '#eab308',
        }.get(s['status'], '#666')

        # Build person cards
        person_html = ""
        for p in persons:
            icon = self.STATUS_ICONS.get(p['status'], '❓')
            label = self.STATUS_LABELS.get(p['status'], p['status'])
            color = {'protected': '#22c55e', 'partial_risk': '#eab308', 'unprotected': '#ef4444'}.get(p['status'], '#666')

            problems_list = ""
            if p['problem_timestamps']:
                problems_list = "<ul>" + "".join(
                    f"<li>{pt['timestamp_str']} (帧 {pt['frame_idx']})</li>"
                    for pt in p['problem_timestamps'][:10]
                ) + "</ul>"
                if len(p['problem_timestamps']) > 10:
                    problems_list += f"<p>... 还有 {len(p['problem_timestamps']) - 10} 处</p>"

            person_html += f"""
            <div class="person-card" style="border-left: 4px solid {color}; padding: 12px; margin: 8px 0; background: #f9f9f9; border-radius: 4px;">
                <h4 style="margin: 0 0 8px 0;">{icon} 人物 #{p['track_id']} - {label}</h4>
                <p style="margin: 4px 0; color: #666; font-size: 14px;">
                    {p['frames_protected']}/{p['frames_total']} 帧已打码
                    | 首次出现: {p['first_appearance']}
                    | 最后出现: {p['last_appearance']}
                </p>
                {problems_list}
            </div>
            """

        # Build screenshot gallery
        screenshots_html = ""
        if screenshot_paths:
            screenshots_html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; margin-top: 20px;">'
            for path in screenshot_paths:
                if os.path.exists(path):
                    try:
                        import base64
                        with open(path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                        filename = os.path.basename(path)
                        screenshots_html += f"""
                        <div style="border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
                            <img src="data:image/jpeg;base64,{img_data}" style="width: 100%; display: block;" />
                            <p style="margin: 0; padding: 8px; font-size: 12px; color: #666; text-align: center;">{filename}</p>
                        </div>
                        """
                    except Exception:
                        pass
            screenshots_html += '</div>'

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>打码检测报告 - {v['filename']}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .container {{ background: white; border-radius: 12px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h1 {{ margin-top: 0; }}
        .status-banner {{ padding: 20px; border-radius: 8px; color: white; text-align: center; font-size: 24px; font-weight: bold; margin: 20px 0; background: {status_color}; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin: 20px 0; }}
        .info-item {{ background: #f9f9f9; padding: 12px; border-radius: 6px; }}
        .info-label {{ color: #666; font-size: 12px; }}
        .info-value {{ font-size: 18px; font-weight: bold; margin-top: 4px; }}
        h2 {{ border-bottom: 2px solid #eee; padding-bottom: 8px; margin-top: 32px; }}
        .screenshot {{ border: 2px dashed #ef4444; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 视频打码质量检测报告</h1>

        <div class="status-banner">
            {'✅ 通过 - 未发现明显问题' if s['status'] == 'passed' else '⚠️ 发现问题 - 请检查以下帧'}
        </div>

        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">文件名</div>
                <div class="info-value">{v['filename']}</div>
            </div>
            <div class="info-item">
                <div class="info-label">时长</div>
                <div class="info-value">{v['duration_str']}</div>
            </div>
            <div class="info-item">
                <div class="info-label">分辨率</div>
                <div class="info-value">{v['width']}x{v['height']}</div>
            </div>
            <div class="info-item">
                <div class="info-label">风险等级</div>
                <div class="info-value">{s['risk_level'].upper()}</div>
            </div>
            <div class="info-item">
                <div class="info-label">检测帧数</div>
                <div class="info-value">{v['sampled_frames']}</div>
            </div>
            <div class="info-item">
                <div class="info-label">问题帧</div>
                <div class="info-value" style="color: {'#22c55e' if s['problem_frames_count'] == 0 else '#ef4444'};">
                    {s['problem_frames_count']} ({s['problem_frames_ratio']*100:.2f}%)
                </div>
            </div>
        </div>

        <h2>👥 人物状态详情</h2>
        {person_html}

        <h2>📸 问题帧截图</h2>
        {screenshots_html if screenshots_html else '<p style="color: #666;">未发现需要截图的问题帧。</p>'}

        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #999; font-size: 12px; text-align: center;">
            生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 隐私视频自检工具
        </div>
    </div>
</body>
</html>"""
        return html

    def _generate_recommendations(self, persons: List[Dict]) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        unprotected = [p for p in persons if p['status'] == 'unprotected']
        partial = [p for p in persons if p['status'] == 'partial_risk']

        if unprotected:
            ids = ', '.join(f"#{p['track_id']}" for p in unprotected)
            recs.append(f"人物 {ids} 全程未打码，请确认是否为有意保留。")

        if partial:
            for p in partial:
                count = len(p['problem_timestamps'])
                recs.append(f"人物 #{p['track_id']} 有 {count} 处漏打，建议检查打码跟踪设置或增加打码强度。")

        if not unprotected and not partial:
            recs.append("所有人物打码完好，建议再做一次人工抽查确认。")

        return recs
