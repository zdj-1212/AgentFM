"""
文本切分器
==========
RAG 的第一步：把长文档切成"适合向量化 + 检索"的小块。

切分原则（企业实践）：
1. 语义完整：优先按段落切，不在句子中间硬切；
2. 控制大小：每块约 300~500 字，兼顾召回精度与上下文容量；
3. 保留重叠：相邻块保留 overlap 字符，避免检索时信息落在切缝处丢失。
"""
from typing import List


def chunk_text(text:str, chunk_size: int = 400, overlap: int=60)->List[str]:
    """
        按段落切分文本，返回若干文本块。

        实现思路：
        - 以换行符把文本拆成段落；
        - 顺序累积段落，直到累积长度 >= chunk_size 就切出一块；
        - 下一块的开头带上上一块的结尾 overlap 个字符，保证连续性。
    """
    if not text or not text.strip():
        return []
    # 1) 按空行/换行拆段落，去掉空段落
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks:List[str] =[]
    buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if buffer:
                chunks.append(buffer)
                buffer=""
            for i in range(0,len(para),chunk_size - overlap):
                chunks.append(para[i : i+chunk_size])
            continue

        if len(buffer) + len(para) <= chunk_size:
            buffer =f"{buffer}\n{para}" if buffer else para
        else:
            chunks.append(buffer)
            buffer = para if len(para) > overlap else (buffer[-overlap:] + "\n" +para)

    if buffer:
        chunks.append(buffer)

    return [c for c in chunks if c.strip()]

def build_chunk_meta(source: str, title: str, chunk_index: int, text: str) -> dict:
    """把切分结果包装成 Milvus 插入所需的文档字典（统一入口）。"""
    return {
        "text": text,
        "title": title,
        "source": source,
        "chunk_index": chunk_index,
    }
