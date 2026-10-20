# -*- coding: utf-8 -*-
"""部门路由：仅部门列表（admin CRUD 预留）。"""

from fastapi import APIRouter, Depends

from app.core.deps import CurrentUser, get_current_user
from app.infra.org.mock_org import get_department_repository

router = APIRouter(prefix="/depts", tags=["depts"])


@router.get("")
async def list_depts(current: CurrentUser = Depends(get_current_user)) -> list[dict]:
    """部门列表（登录即可访问，上传文档选部门要用）。"""
    repo = get_department_repository()
    depts = await repo.all()
    return [{"id": d.id, "code": d.code, "name": d.name} for d in depts]
