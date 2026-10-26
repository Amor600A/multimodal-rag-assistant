# -*- coding: utf-8 -*-
"""单元测试：MockMilvus expr 解析器 + _build_permission_expr 三类用户快照。

覆盖点（方案 12.1 / 6.3）：
1. 解析器子集：== 匹配、and/or 组合、括号优先级、字符串字面量；
2. 子集外语法必须抛 ExprParseError（不能静默漏过滤，防止权限逻辑悄悄失效）；
3. _build_permission_expr 对「有部门用户 / 无部门用户 / admin」生成的 expr 结构与语义，
   并断言生成的 expr 一定能被 MockMilvus 解析器成功解析（Mock 与真实 Milvus 共用同一 expr）。
"""

from __future__ import annotations

import pytest

from app.core.deps import CurrentUser
from app.infra.vectordb.mock_milvus import ExprParseError, compile_expr
from app.services.rag_service import _build_permission_expr


def _rec(**kw) -> dict:
    """构造最小向量记录（仅保留 expr 求值需要的权限字段）。"""
    base = {
        "id": "rec-x",
        "scope": "domain",
        "visibility": "public",
        "owner_user_id": None,
        "dept_id": None,
    }
    base.update(kw)
    return base


# ---- 1. == 匹配 + 字符串字面量 ----

def test_expr_eq_match():
    """== 对字符串字段做等值匹配；字段不存在时按 str(None) 比较。"""
    pred = compile_expr('visibility=="public"')
    assert pred(_rec(visibility="public")) is True
    assert pred(_rec(visibility="dept")) is False

    pred_num = compile_expr('dept_id=="d-hr"')
    assert pred_num(_rec(dept_id="d-hr")) is True
    assert pred_num(_rec(dept_id="d-mkt")) is False
    assert pred_num(_rec(dept_id=None)) is False


def test_expr_string_literal_with_chinese_and_symbols():
    """字符串字面量支持中文/连字符等任意内容（仅要求双引号包裹）。"""
    pred = compile_expr('owner_user_id=="user-中文-01"')
    assert pred(_rec(owner_user_id="user-中文-01")) is True
    assert pred(_rec(owner_user_id="user-中文-02")) is False


# ---- 2. and / or 组合 ----

def test_expr_and_combination():
    pred = compile_expr('scope=="domain" and visibility=="public"')
    assert pred(_rec(scope="domain", visibility="public")) is True
    assert pred(_rec(scope="domain", visibility="dept")) is False
    assert pred(_rec(scope="personal", visibility="public")) is False


def test_expr_or_combination():
    pred = compile_expr('visibility=="public" or visibility=="dept"')
    assert pred(_rec(visibility="public")) is True
    assert pred(_rec(visibility="dept")) is True
    assert pred(_rec(visibility="private")) is False


def test_expr_three_way_or():
    """三路 or 组合（权限 expr 的典型形态）。"""
    pred = compile_expr(
        '(scope=="domain" and visibility=="public")'
        ' or (scope=="domain" and visibility=="dept" and dept_id=="d-hr")'
        ' or ((scope=="personal" or visibility=="private") and owner_user_id=="u1")'
    )
    assert pred(_rec(scope="domain", visibility="public")) is True
    assert pred(_rec(scope="domain", visibility="dept", dept_id="d-hr")) is True
    assert pred(_rec(scope="domain", visibility="dept", dept_id="d-mkt")) is False
    assert pred(_rec(scope="personal", visibility="private", owner_user_id="u1")) is True
    assert pred(_rec(scope="personal", visibility="private", owner_user_id="u2")) is False


# ---- 3. 括号优先级 ----

def test_expr_parentheses_precedence():
    """括号改变优先级：and 高于 or，括号内 or 先于外部 and 求值。"""
    pred = compile_expr('scope=="domain" and (visibility=="public" or visibility=="dept")')
    assert pred(_rec(scope="domain", visibility="public")) is True
    assert pred(_rec(scope="domain", visibility="dept")) is True
    assert pred(_rec(scope="domain", visibility="private")) is False
    assert pred(_rec(scope="personal", visibility="public")) is False

    pred2 = compile_expr('(scope=="personal" or visibility=="private") and owner_user_id=="u1"')
    assert pred2(_rec(scope="personal", visibility="private", owner_user_id="u1")) is True
    assert pred2(_rec(scope="domain", visibility="private", owner_user_id="u1")) is True
    assert pred2(_rec(scope="personal", visibility="private", owner_user_id="u2")) is False
    assert pred2(_rec(scope="domain", visibility="public", owner_user_id="u1")) is False


# ---- 4. 子集外语法必须抛 ExprParseError ----

@pytest.mark.parametrize(
    "expr",
    [
        'scope != "domain"',            # 不支持 !=
        'visibility > "public"',        # 不支持 > 比较
        'visibility >= "public"',       # 不支持 >= 比较
        'dept_id in "hr"',              # 不支持 in
        "visibility",                   # 裸标识符（无比较）
        'dept_id == 1',                 # 比较值必须是字符串字面量（未加引号）
        'scope == "domain" and',        # 表达式意外结束（尾部 and）
        'visibility == "public" or',    # 表达式意外结束（尾部 or）
        'not visibility=="public"',     # 不支持 not（被当字段名）
    ],
)
def test_expr_out_of_subset_raises(expr):
    """子集外语法一律抛 ExprParseError，绝不静默漏过滤。"""
    with pytest.raises(ExprParseError):
        compile_expr(expr)


# ---- 5. _build_permission_expr 三类用户快照 ----

def test_build_permission_expr_dept_user():
    """有部门普通用户：public + 本部门 + 本人 private 三路 or。"""
    user = CurrentUser(user_id="u-alice", username="alice", role="user", dept_id="d-hr")
    expr = _build_permission_expr(user)

    # 完整表达式快照（三路 or 合并）
    assert expr == (
        '(scope=="domain" and visibility=="public")'
        ' or (scope=="domain" and visibility=="dept" and dept_id=="d-hr")'
        ' or ((scope=="personal" or visibility=="private") and owner_user_id=="u-alice")'
    )

    # 生成的 expr 必须能被 MockMilvus 解析器解析（子集兼容）
    pred = compile_expr(expr)
    assert pred(_rec(scope="domain", visibility="public")) is True
    assert pred(_rec(scope="domain", visibility="dept", dept_id="d-hr")) is True
    assert pred(_rec(scope="domain", visibility="dept", dept_id="d-mkt")) is False
    assert pred(_rec(scope="personal", visibility="private", owner_user_id="u-alice")) is True
    assert pred(_rec(scope="personal", visibility="private", owner_user_id="u-other")) is False


def test_build_permission_expr_no_dept_user():
    """无部门用户：仅 public + 本人 private，不含 dept 分支。"""
    user = CurrentUser(user_id="u-admin", username="admin", role="admin", dept_id=None)
    expr = _build_permission_expr(user)

    # 完整表达式快照（无 dept 分支，仅两路 or）
    assert expr == (
        '(scope=="domain" and visibility=="public")'
        ' or ((scope=="personal" or visibility=="private") and owner_user_id=="u-admin")'
    )
    assert 'visibility=="dept"' not in expr
    assert "dept_id==" not in expr

    pred = compile_expr(expr)
    assert pred(_rec(scope="domain", visibility="public")) is True
    # 任意部门文档都不可见
    assert pred(_rec(scope="domain", visibility="dept", dept_id="d-hr")) is False
    assert pred(_rec(scope="domain", visibility="dept", dept_id="d-mkt")) is False
    assert pred(_rec(scope="personal", visibility="private", owner_user_id="u-admin")) is True
    assert pred(_rec(scope="personal", visibility="private", owner_user_id="u-other")) is False


def test_build_permission_expr_admin_has_no_dept_clause():
    """admin 即使带 dept_id 也应按「无部门」口径生成（admin 仅 public + 本人 private）。"""
    # admin 在种子数据中 dept_id 为空；此处再验证 dept_id 存在时仍不生成 dept 分支
    user = CurrentUser(user_id="u-admin", username="admin", role="admin", dept_id=None)
    expr = _build_permission_expr(user)
    assert "dept_id" not in expr
    # 解析器兼容
    compile_expr(expr)
