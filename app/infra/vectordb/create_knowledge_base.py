# -*- coding: utf-8 -*-
"""创建 Milvus Lite 知识库脚本。

流程：读取 PDF/DOCX -> 提取文本+页码 -> 切片 -> 嵌入 -> 写入 Milvus Lite。
数据文件默认落在 data/milvus/rag_knowledge.db（由配置 MILVUS_DB_PATH 决定）。

支持格式：
    .pdf  -> 逐页提取，页码精确到页
    .docx -> 无固定分页（分页由渲染决定），页码统一记为 1
    .doc  -> Word 97-2003 二进制文档，经 Word COM 转成 .docx 后按 .docx 规则提取，页码记为 1
    .txt  -> 纯文本，UTF-8 优先、GB18030 兜底，无分页，页码记为 1
    .jpg/.jpeg/.png/.bmp -> 图片不切分，一个文件一条记录（需 EMBEDDING_BACKEND=dashscope_mm）
    .mp4/.mov/.avi/.webm（公网 URL）-> 视频不切分，一条记录，同样需要多模态嵌入

图片/视频记录没有正文，text 字段存 --description（缺省用文件名），
用于检索结果展示与关键词过滤。

改造说明（相对最初参考脚本）：
1. 嵌入改用项目统一嵌入器（EMBEDDING_BACKEND=dashscope_mm 时为
   DashScope multimodal-embedding-v1，文本/图片/视频同处一个 1024 维语义空间），
   与线上检索共用同一向量空间
2. 存储改用项目 MilvusLite（接口与 MockMilvus 一致），不再依赖 langchain_milvus
3. 每条切片补充权限标量字段（scope/visibility/owner_user_id/dept_id），
   否则检索时的权限 expr 无法命中
4. 页码写入 source 字段（格式：文件名#p页码），不额外占用 schema 字段

用法：
    python -m app.infra.vectordb.create_knowledge_base 文档.pdf
    python -m app.infra.vectordb.create_knowledge_base 文档.docx --visibility dept --dept-id <部门ID>
    python -m app.infra.vectordb.create_knowledge_base 文档.doc --visibility dept --dept-id <部门ID>
    python -m app.infra.vectordb.create_knowledge_base 手册.txt
    python -m app.infra.vectordb.create_knowledge_base 文档.pdf --visibility private --owner-user-id <用户ID>
    python -m app.infra.vectordb.create_knowledge_base 图片.png --description "园区导览图"
    python -m app.infra.vectordb.create_knowledge_base https://example.com/a.mp4 --description "入园须知视频"

注意：Milvus Lite 为单文件独占，运行本脚本前请先停止后端服务，避免文件被占用。
      .doc 需要本机安装 Microsoft Word（通过 COM 转换格式），.txt 需为 UTF-8 或 GBK 编码。
"""

from __future__ import annotations

import argparse
import hashlib
import os
from typing import List, Tuple

from pypdf import PdfReader

# 图片格式：本地文件，走 Base64 上传
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# 视频格式：仅支持公网 URL
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm"}

# 默认切片参数（与参考脚本保持一致）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# 分隔符优先级：段落 -> 换行 -> 中文句末 -> 英文句点 -> 空格
_SEPARATORS = ("\n\n", "\n", "。", "！", "？", ".", " ")


def extract_text_from_pdf(pdf) -> Tuple[str, List[int]]:
    """从 PDF 提取文本，并记录每个字符对应的页码。

    参数:
        pdf: PdfReader 对象
    返回:
        text: 提取的文本内容
        char_page_mapping: 与 text 等长的页码列表
    """
    text = ""
    char_page_mapping: list[int] = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            # 为当前页面的每个字符记录页码
            char_page_mapping.extend([page_number] * len(extracted_text))
        else:
            print(f"第 {page_number} 页未提取到文本，已跳过。")

    return text, char_page_mapping

def extract_text_from_docx(docx_path: str) -> Tuple[str, List[int]]:
    """从 DOCX 提取文本。

    DOCX 没有固定分页（分页由渲染时决定），无法逐页定位，
    因此页码统一记为 1，source 溯源表现为 文件名#p1。

    提取范围：正文段落 + 表格单元格文本（攻略/政策类文档常用表格排版）。

    参数:
        docx_path: DOCX 文件路径
    返回:
        text: 提取的文本内容
        char_page_mapping: 与 text 等长的页码列表（全部为 1）
    """
    from docx import Document  # 延迟导入：仅处理 DOCX 时要求 python-docx

    document = Document(docx_path)
    parts: list[str] = []
    for para in document.paragraphs:
        t = para.text.strip()
        if t:
            parts.append(t)
    for table in document.tables:
        for row in table.rows:
            # 横向合并单元格（gridSpan）会让 row.cells 把同一个单元格返回多次，
            # 按底层 XML 节点去重；不能用文本去重，因为两列内容相同是合法的
            cells: list[str] = []
            seen: set = set()
            for cell in row.cells:
                if cell._tc in seen:
                    continue
                seen.add(cell._tc)
                t = cell.text.strip()
                if t:
                    cells.append(t)
            line = " | ".join(cells)
            if line:
                parts.append(line)

    text = "\n".join(parts)
    return text, [1] * len(text)

def extract_text_from_doc(doc_path: str) -> Tuple[str, List[int]]:
    """从旧版 Word 文档（.doc，Word 97-2003）提取文本。

    .doc 是 OLE2 复合二进制文档，python-docx 无法直接读取，
    这里用 Word COM 另存为 .docx 后复用 extract_text_from_docx，
    保证 .doc 与 .docx 的提取规则（正文段落 + 表格单元格）完全一致。

    .doc 的分页同样由渲染决定，页码统一记为 1。

    参数:
        doc_path: DOC 文件路径
    返回:
        text: 提取的文本内容
        char_page_mapping: 与 text 等长的页码列表（全部为 1）
    """
    import shutil
    import tempfile

    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise SystemExit(
            "处理 .doc 需要 pywin32（pip install pywin32），且本机需安装 Microsoft Word"
        ) from exc

    tmp_dir = tempfile.mkdtemp(prefix="rag_doc_")
    docx_path = os.path.join(tmp_dir, "converted.docx")
    pythoncom.CoInitialize()
    word = None
    try:
        word = win32com.client.Dispatch("Word.Application")
        # 后台静默转换：不显示窗口、不弹恢复/覆盖提示
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(
            os.path.abspath(doc_path), ReadOnly=True, AddToRecentFiles=False
        )
        doc.SaveAs2(docx_path, FileFormat=12)  # 12 = wdFormatXMLDocument
        doc.Close(False)
    finally:
        if word is not None:
            word.Quit()
        pythoncom.CoUninitialize()

    try:
        return extract_text_from_docx(docx_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def extract_text_from_txt(txt_path: str) -> Tuple[str, List[int]]:
    """从纯文本文件提取文本。

    编码顺序：UTF-8（含 BOM）优先；中文 Windows 下记事本常另存为 GBK，
    失败时按 GB18030 解码（GBK 的超集，不会静默产出乱码）。

    参数:
        txt_path: 文本文件路径
    返回:
        text: 提取的文本内容
        char_page_mapping: 与 text 等长的页码列表（全部为 1）
    """
    with open(txt_path, "rb") as f:
        raw = f.read()

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("gb18030")

    return text, [1] * len(text)


def _media_embedder():
    """取出多模态嵌入器；图片/视频入库必须使用统一多模态空间。"""
    from app.core.config import get_settings
    from app.rag.embedder import MultimodalEmbedder, get_embedder

    embedder = get_embedder()
    if not isinstance(embedder, MultimodalEmbedder):
        raise SystemExit(
            "图片/视频入库需要 EMBEDDING_BACKEND=dashscope_mm，"
            f"当前为 {get_settings().embedding_backend}"
        )
    return embedder


def embed_image(image_path: str):
    """生成图片向量（1024 维，与文本同一语义空间）。

    由嵌入器负责读取本地文件并转 Base64，支持 jpg/jpeg/png/bmp。
    """
    return _media_embedder().embed_image(image_path)


def embed_video(video_url: str):
    """生成视频向量（1024 维，与文本同一语义空间）。

    接口只接受公网可访问的 http(s) URL，本地视频文件需先上传到可访问地址。
    单条视频返回一个向量，不做多帧平均。
    """
    return _media_embedder().embed_video(video_url)

def _split_with_offsets(text: str) -> List[Tuple[str, int]]:
    """按语义边界切片，并返回每块在原文中的起始偏移。

    与直接按长度硬切相比，偏移量让页码映射可以精确定位到原文位置，
    不会因 chunk_overlap 导致字符位置漂移。

    返回:
        [(chunk_text, start_offset), ...]
    """
    chunks: list[tuple[str, int]] = []
    total = len(text)
    start = 0

    while start < total:
        end = min(start + CHUNK_SIZE, total)
        if end < total:
            # 只在窗口后半段找边界，避免切出过短的块
            window = text[start:end]
            cut = -1
            for sep in _SEPARATORS:
                idx = window.rfind(sep)
                if idx != -1 and idx >= CHUNK_SIZE // 2:
                    cut = idx + len(sep)
                    break
            if cut > 0:
                end = start + cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start))

        if end >= total:
            break
        # 向前回退 overlap 个字符，保持块间上下文连续
        start = max(end - CHUNK_OVERLAP, start + 1)

    print(f"文本被分割成 {len(chunks)} 个块。")
    return chunks

def split_text(text: str) -> List[str]:
    """兼容原接口：只返回切片文本列表。"""
    return [chunk for chunk, _ in _split_with_offsets(text)]

def compute_chunk_pages(
    chunks: List[Tuple[str, int]], char_page_mapping: List[int]
) -> dict:
    """为每个文本块计算主要来源页码（取块内字符页码的众数）。

    参数:
        chunks: [(chunk_text, start_offset), ...]
        char_page_mapping: 每个字符对应的页码列表
    返回:
        page_info: {文本块内容: 页码}
    """
    print("\n========== 预处理切分的全部文本块 ==========")

    page_info: dict[str, int] = {}
    for i, (chunk, offset) in enumerate(chunks, start=1):
        # 按块在原文中的位置取对应页码范围
        page_slice = char_page_mapping[offset:offset + len(chunk)]
        if page_slice:
            page_counts: dict[int, int] = {}
            for page in page_slice:
                page_counts[page] = page_counts.get(page, 0) + 1
            most_common_page = max(page_counts.items(), key=lambda kv: kv[1])[0]
        else:
            most_common_page = 1

        page_info[chunk] = most_common_page

        print(f"\n--- 第 {i} 个文本块 (长度: {len(chunk)} 字符, 页码: {most_common_page}) ---")
        print(chunk)

    print("\n========== 预处理切分的全部文本块打印结束 ==========\n")
    print(f"页码映射完成，共 {len(page_info)} 个文本块")
    return page_info


def build_milvus(
    chunks: List[str],
    page_info: dict,
    doc_name: str,
    save_path: str,
    visibility: str,
    dept_id: str | None,
    owner_user_id: str | None,
):
    """把文本块与权限标量写入 Milvus Lite。

    参数:
        chunks: 文本块列表
        page_info: {文本块内容: 页码}
        doc_name: 来源文档名（写入 source 字段）
        save_path: Milvus Lite 数据文件路径
        visibility: public | dept | private
        dept_id: visibility=dept 时的部门 ID
        owner_user_id: visibility=private 时的归属用户 ID
    返回:
        已写入的 MilvusLite 实例
    """
    from app.core.config import get_settings
    from app.infra.vectordb.milvus_lite import MilvusLite

    settings = get_settings()
    db = MilvusLite(
        db_path=save_path,
        collection_name=settings.milvus_collection,
        dim=settings.embedding_dim,
    )

    # private 归入 personal 区，其余归入 domain 区（与 knowledge 路由的打标规则一致）
    scope = "personal" if visibility == "private" else "domain"

    records: list[dict] = []
    for i, chunk in enumerate(chunks):
        page = page_info[chunk]
        # 用内容哈希做 id：同一份 PDF 重复建库时幂等覆盖，不会产生重复块
        digest = hashlib.sha1(f"{doc_name}|{page}|{chunk}".encode("utf-8")).hexdigest()[:16]
        records.append({
            "id": f"pdf-{digest}",
            "text": chunk,
            "scope": scope,
            "visibility": visibility,
            "owner_user_id": owner_user_id,
            "dept_id": dept_id,
            "source": f"{doc_name}#p{page}",
        })

    db.insert(records)
    print(f"已写入 {len(records)} 条切片，数据文件: {save_path}")
    print(f"当前集合记录总数: {db.count()}（collection={settings.milvus_collection}）")
    print(f"嵌入后端: {settings.embedding_backend}（model={settings.embedding_model}）")
    return db


def build_media(
    vector,
    description: str,
    doc_name: str,
    save_path: str,
    visibility: str,
    dept_id: str | None,
    owner_user_id: str | None,
    id_prefix: str,
):
    """写入单条媒体记录：图片/视频不切分，一个文件对应一个向量。

    参数:
        vector: 多模态嵌入向量（1024 维）
        description: 写入 text 字段的说明文本（检索结果展示用）
        doc_name: 来源文件名或视频 URL（写入 source 字段）
        save_path: Milvus Lite 数据文件路径
        visibility: public | dept | private
        dept_id: visibility=dept 时的部门 ID
        owner_user_id: visibility=private 时的归属用户 ID
        id_prefix: 主键前缀，图片 img、视频 vid
    返回:
        已写入的 MilvusLite 实例
    """
    from app.core.config import get_settings
    from app.infra.vectordb.milvus_lite import MilvusLite

    settings = get_settings()
    db = MilvusLite(
        db_path=save_path,
        collection_name=settings.milvus_collection,
        dim=settings.embedding_dim,
    )

    scope = "personal" if visibility == "private" else "domain"
    digest = hashlib.sha1(f"{doc_name}|{description}".encode("utf-8")).hexdigest()[:16]
    records = [{
        "id": f"{id_prefix}-{digest}",
        "vector": vector,
        "text": description,
        "scope": scope,
        "visibility": visibility,
        "owner_user_id": owner_user_id,
        "dept_id": dept_id,
        "source": doc_name,
    }]

    db.insert(records)
    print(f"已写入 1 条媒体记录（不切分），数据文件: {save_path}")
    print(f"当前集合记录总数: {db.count()}（collection={settings.milvus_collection}）")
    print(f"嵌入后端: {settings.embedding_backend}（model={settings.embedding_model}）")
    return db


def process_text_with_splitter(
    text: str,
    char_page_mapping: List[int],
    doc_name: str,
    save_path: str,
    visibility: str = "public",
    dept_id: str | None = None,
    owner_user_id: str | None = None,
):
    """处理文本并创建向量存储（保留原函数名与流程）。"""
    chunks = _split_with_offsets(text)
    page_info = compute_chunk_pages(chunks, char_page_mapping)
    return build_milvus(
        [chunk for chunk, _ in chunks],
        page_info,
        doc_name,
        save_path,
        visibility,
        dept_id,
        owner_user_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="创建 Milvus Lite 知识库")
    parser.add_argument(
        "file",
        help="文档路径（.pdf/.docx/.doc/.txt）或图片路径（.jpg/.jpeg/.png/.bmp），视频传 http(s) URL",
    )
    parser.add_argument("--visibility", default="public", choices=["public", "dept", "private"])
    parser.add_argument("--dept-id", default=None, help="visibility=dept 时必填")
    parser.add_argument("--owner-user-id", default=None, help="visibility=private 时必填")
    parser.add_argument("--db-path", default=None, help="覆盖 MILVUS_DB_PATH")
    parser.add_argument("--description", default=None, help="图片/视频的说明文本，缺省用文件名")
    args = parser.parse_args()

    if args.visibility == "dept" and not args.dept_id:
        raise SystemExit("visibility=dept 时必须提供 --dept-id")
    if args.visibility == "private" and not args.owner_user_id:
        raise SystemExit("visibility=private 时必须提供 --owner-user-id")

    # 延迟导入配置，确保存在 .env 时能正确读取嵌入配置
    from app.core.config import get_settings

    settings = get_settings()
    db_path = args.db_path or settings.milvus_db_path

    is_url = args.file.lower().startswith(("http://", "https://"))
    ext = os.path.splitext(args.file)[1].lower()

    # 图片：本地文件，一条记录，不切片
    if not is_url and ext in _IMAGE_EXTS:
        doc_name = os.path.basename(args.file)
        description = args.description or doc_name
        vector = embed_image(args.file)
        build_media(
            vector, description, doc_name, db_path,
            args.visibility, args.dept_id, args.owner_user_id, "img",
        )
        print(f"\n知识库构建完成，数据文件: {db_path}")
        return

    # 视频：仅公网 URL，一条记录，不切片
    if is_url and ext in _VIDEO_EXTS:
        description = args.description or os.path.basename(args.file)
        vector = embed_video(args.file)
        build_media(
            vector, description, args.file, db_path,
            args.visibility, args.dept_id, args.owner_user_id, "vid",
        )
        print(f"\n知识库构建完成，数据文件: {db_path}")
        return

    if ext == ".pdf":
        pdf_reader = PdfReader(args.file)
        text, char_page_mapping = extract_text_from_pdf(pdf_reader)
    elif ext == ".docx":
        text, char_page_mapping = extract_text_from_docx(args.file)
    elif ext == ".doc":
        text, char_page_mapping = extract_text_from_doc(args.file)
    elif ext == ".txt":
        text, char_page_mapping = extract_text_from_txt(args.file)
    else:
        raise SystemExit(
            f"暂不支持的文件类型: {ext}（支持 .pdf/.docx/.doc/.txt，"
            "图片 .jpg/.jpeg/.png/.bmp，视频以 http(s) URL 传入）"
        )
    print(f"提取的文本长度: {len(text)} 个字符。")

    doc_name = os.path.basename(args.file)
    process_text_with_splitter(
        text,
        char_page_mapping,
        doc_name,
        db_path,
        args.visibility,
        args.dept_id,
        args.owner_user_id,
    )
    print(f"\n知识库构建完成，数据文件: {db_path}")


if __name__ == "__main__":
    main()
