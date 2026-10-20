# -*- coding: utf-8 -*-
"""Mock LLM：按关键词匹配返回预设回复。

当前阶段无真实 LLM，所有响应来自 canned 文案。
阶段 4 接真实 LLM 时替换此模块即可，chat SSE 路由和 respond_node 不动。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("app.agent.mock_llm")


# 预设响应对话（按关键词做简陋匹配，够阶段 3 验收即可）
_CANNED_REPLIES: dict[str, str] = {
    "考勤": "关于考勤请假，根据公司人事制度，正式员工每月有 2 天带薪病假。请在 OA 系统提交请假申请并上传医院证明。",
    "请假": "请假流程：OA 提交申请 → 直属主管审批 → 部门 HR 确认。病假需附医院证明，事假提前 3 个工作日。",
    "病假": "病假需附县级以上医院证明，OA 提交后 24 小时内直属主管审批，超期未审批自动流转部门 HR。",
    "竞品": "竞品分析模板已发送到您的企业微信，请查收附件。关键维度含：产品定位、定价策略、核心功能、市场份额、SWOT。",
    "市场": "市场部 2025 Q4 预算已过会，整体较 Q3 上浮 12%，重点投向抖音信息流与私域运营。",
}
_FALLBACK = "我是多模态 RAG 助手（Mock 模式）。当前阶段仅支持基础对话，暂未接入知识库检索与工具调用。"


def pick_reply(user_text: str) -> str:
    """按关键词匹配预设回复，命中多个取最长匹配。"""
    best = ""
    for kw, reply in _CANNED_REPLIES.items():
        if kw in user_text and len(reply) > len(best):
            best = reply
    return best or _FALLBACK
