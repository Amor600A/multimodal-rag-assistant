# -*- coding: utf-8 -*-
"""Chat SSE 集成测试：帧序列、终态互斥、warning 净化、会话归属、并发单飞。

SSE 读取：with client.stream("POST", "/api/chat", ...) 逐行读，
data: 前缀行是 JSON 帧（帧类型见方案 4.1）。
"""

from __future__ import annotations

import json
import threading

from langchain_core.messages import AIMessage


def _read_sse(client, headers: dict, payload: dict) -> list[dict]:
    """发起一次 chat SSE 请求并返回全部数据帧（按 data: 行解析）。"""
    frames: list[dict] = []
    with client.stream("POST", "/api/chat", json=payload, headers=headers) as resp:
        assert resp.status_code == 200, resp.read()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            frames.append(json.loads(line[len("data:"):].strip()))
    return frames


def _token_text(frames: list[dict]) -> str:
    """拼接全部 token 帧得到完整回复正文。"""
    return "".join(f["content"] for f in frames if f["type"] == "token")


def _types(frames: list[dict]) -> list[str]:
    return [f["type"] for f in frames]


# ---- 1. 正常对话：帧序列 + token 拼接 + 来源引用 ----

def test_normal_chat_sse_sequence(client, headers_alice):
    frames = _read_sse(
        client,
        headers_alice,
        {"message": "考勤请假流程是怎样的？", "thread_id": "seed-alice-1"},
    )

    types = _types(frames)
    # 帧序列包含 node / token / done
    assert "node" in types
    assert "token" in types
    assert "done" in types

    # node 帧覆盖链路三个节点
    nodes = {f["node"] for f in frames if f["type"] == "node"}
    assert {"retrieve", "generate", "output_guard"} <= nodes

    # 拼接 token 得到完整回复，检索命中时末尾带来源引用行
    content = _token_text(frames)
    assert content.strip()
    assert "来源:" in content
    assert "来源: [1]" in content


# ---- 2. 终态互斥：一条流只出现一个终态帧 ----

def test_terminal_frame_mutual_exclusion(client, headers_alice):
    frames = _read_sse(
        client,
        headers_alice,
        {"message": "考勤请假流程是怎样的？", "thread_id": "seed-alice-1"},
    )
    terminals = [t for t in _types(frames) if t in ("done", "error")]
    # 只出现一个终态帧，不会既 error 又 done（方案 4.1 终态互斥）
    assert len(terminals) == 1
    assert terminals == ["done"]


# ---- 3. warning 帧：伪造引用被 output_guard 净化 ----

def test_warning_frame_for_fake_citation(client, headers_alice, monkeypatch):
    """monkeypatch generate 节点产出伪造引用 [99]，断言 warning 帧且正文不含 [99]。

    说明：graph 编译后单例缓存对节点函数的引用，直接改函数不会生效，
    因此需同时重置 app.agent.graph._compiled 触发重新编译；
    monkeypatch 会在用例结束自动还原两者。
    """
    import app.agent.graph as graph_mod

    async def _fake_generate(state):
        # 伪造末尾引用编号 [99]（不在召回集合内），触发 output_guard 净化
        reply = "这是测试生成的正文内容。"
        reply = f"{reply}\n\n来源: [99]"
        return {"messages": [AIMessage(content=reply)]}

    monkeypatch.setattr(graph_mod, "generate_node", _fake_generate)
    monkeypatch.setattr(graph_mod, "_compiled", None)

    frames = _read_sse(
        client,
        headers_alice,
        {"message": "考勤请假流程是怎样的？", "thread_id": "seed-alice-1"},
    )

    warnings = [f for f in frames if f["type"] == "warning"]
    assert warnings, frames
    assert "99" in warnings[0]["detail"]

    content = _token_text(frames)
    assert "这是测试生成的正文内容" in content
    assert "[99]" not in content


# ---- 4. 会话归属：不存在的 thread_id -> error 帧（非 500）----

def test_unknown_thread_returns_error_frame(client, headers_alice):
    frames = _read_sse(
        client,
        headers_alice,
        {"message": "你好", "thread_id": "not-exist-thread-xyz"},
    )
    types = _types(frames)
    assert types.count("error") == 1
    assert "done" not in types


# ---- 5. 并发保护：同用户同 thread_id 同时发两个请求 ----

def test_concurrent_same_thread_requests(client, headers_alice):
    """当前实现用 asyncio.Lock 排队等待（非返回 409，与方案 12.3 描述不一致）。

    预期：两个请求都正常完成、无异常、各自只出现一个 done 终态帧。
    """
    payload = {"message": "考勤请假流程是怎样的？", "thread_id": "seed-alice-1"}
    results: list[list[dict] | None] = [None, None]
    errors: list[BaseException] = []

    def worker(idx: int) -> None:
        try:
            results[idx] = _read_sse(client, headers_alice, payload)
        except BaseException as e:  # noqa: BLE001 - 记录线程内异常供主线程断言
            errors.append(e)

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=60)
    t2.join(timeout=60)

    assert not errors, errors
    assert results[0] and results[1]
    for frames in results:
        terminals = [t for t in _types(frames) if t in ("done", "error")]
        assert terminals == ["done"], frames
        # 帧内顺序完整：token 拼接非空
        assert _token_text(frames).strip()
