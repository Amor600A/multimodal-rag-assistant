# -*- coding: utf-8 -*-
"""图片元数据仓储抽象层。"""

from .mock_images import ImageRecord, MockImageRepository, get_image_repository

__all__ = ["ImageRecord", "MockImageRepository", "get_image_repository"]
