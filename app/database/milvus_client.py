"""
Milvus 向量数据库客户端
========================
Milvus 是业界主流的开源向量数据库，本模块封装：
- 集合（Collection）的创建 / 删除 / schema 自检
- 向量写入（入库）
- 稠密向量检索（ANN 近似最近邻）
- 稠密 + BM25 稀疏的**混合检索**与重排

技术要点：
- 使用 pymilvus 的轻量客户端 MilvusClient（比旧版 Collection API 更简洁、推荐）；
- 度量方式用 COSINE。**注意 Milvus 对 COSINE 返回的 distance 就是余弦相似度
  （越大越相似），不是距离**——曾把它当成距离做过 `1 - distance`，结果把相关性
  整个反过来（最相关的被阈值滤掉），这类错误不报错、只让回答质量悄悄变差；
- 检索时用相似度阈值过滤低质量结果，防止"强行回答"；
- 混合检索的稀疏分支由 Milvus 的 BM25 Function 从 text 自动生成，无需自己算。
"""
from functools import lru_cache
from typing import Dict,List,Optional

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    MilvusException,
    RRFRanker,
    WeightedRanker,
)

from config.settings import settings
from app.core.logger import get_logger

logger=get_logger(__name__)

@lru_cache(maxsize=1)
def get_milvus()->MilvusClient:
    """全局唯一 MilvusClient 单例（连接是懒建立 + 缓存的）。"""
    logger.info("连接 Milvus: %s", settings.MILVUS_URI)
    return MilvusClient(uri=settings.MILVUS_URI)

def collection_exists()->bool:
    return get_milvus().has_collection(settings.MILVUS_COLLECTION)

def create_collection()->None:
    """
    创建知识库集合（稠密向量 + BM25 稀疏向量，供混合检索使用）。

    字段设计：
      - id          主键（自增）
      - text        原始文本（召回后直接作为 LLM 上下文）；enable_analyzer 是 BM25 的前提
      - title       所属文档标题
      - source      来源文件/章节（用于引用溯源）
      - chunk_index 第几个切片
      - vector      稠密向量，维度 = embedding 维度
      - sparse      BM25 稀疏向量，**由 Milvus 的 Function 从 text 自动生成**

    注意 sparse 不需要（也不能）在入库时自己填：它是 BM25 Function 的输出字段，
    建好 Function 与索引后，插入 text 时 Milvus 会自动算出稀疏向量。
    """
    dim =settings.EMBEDDING_DIM
    client=get_milvus()
    collection=settings.MILVUS_COLLECTION
    if client.has_collection(collection):
        logger.warning(f"集合 {collection} 已存在，跳过创建")
        return

    schema=client.create_schema(auto_id=True,enable_dynamic_field=False)
    schema.add_field("id",DataType.INT64,is_primary=True,auto_id=True)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=8192,
        # 开启分词后才能被 BM25 Function 索引；中文必须配中文分词器，
        # 否则按空白切分会把整段当成一个词，词面召回形同虚设
        enable_analyzer=True,
        analyzer_params={"type": settings.MILVUS_TEXT_ANALYZER},
    )
    schema.add_field("title",DataType.VARCHAR,max_length=512)
    schema.add_field("source",DataType.VARCHAR,max_length=512)
    schema.add_field("chunk_index",DataType.INT64)
    schema.add_field("vector",DataType.FLOAT_VECTOR,dim=dim)
    schema.add_field("sparse",DataType.SPARSE_FLOAT_VECTOR)

    # 声明"text -> sparse"的 BM25 变换，Milvus 会在写入时自动计算
    schema.add_function(
        Function(
            name="text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"],
        )
    )

    index_params=client.prepare_index_params()
    index_params.add_index(
        field_name="vector",index_type="IVF_FLAT",metric_type="COSINE",params={"nlist": 1024}
    )
    index_params.add_index(
        field_name="sparse",index_type="SPARSE_INVERTED_INDEX",metric_type="BM25"
    )
    client.create_collection(
        collection_name=collection,
        schema=schema,
        index_params=index_params
    )
    logger.info(
        f"Milvus 集合创建完成: {collection} (dim={dim}, 稠密+BM25 稀疏, 分词器={settings.MILVUS_TEXT_ANALYZER})"
    )

def supports_hybrid()->bool:
    """
    当前集合是否具备混合检索所需的 schema（即是否含 sparse 字段）。

    老版本的集合只有稠密向量字段，而 Milvus **不支持给已有集合加 Function/稀疏字段**，
    只能重建。所以这里显式探测，让调用方给出可执行的提示，而不是在检索时报一句
    "field sparse not found" 让人摸不着头脑。
    """
    if not collection_exists():
        return False
    try:
        info = get_milvus().describe_collection(settings.MILVUS_COLLECTION)
        names = {f.get("name") for f in info.get("fields", [])}
        return "sparse" in names
    except Exception as e:
        logger.warning("探测集合 schema 失败: %s", e)
        return False

def drop_collection()->None:
    """删除集合（重新入库前使用，危险操作请谨慎）。"""
    if collection_exists():
        get_milvus().drop_collection(settings.MILVUS_COLLECTION)
        logger.info(f"已删除集合：{settings.MILVUS_COLLECTION}")

def insert_documents(doc:List[Dict])->int:
    """
        批量写入向量数据。
        入参 docs: [{text, title, source, chunk_index, vector}, ...]
        返回写入条数。
    """
    create_collection()
    data=[
        {
            "text":d["text"],
            "title":d["title"],
            "source":d["source"],
            "chunk_index":d["chunk_index"],
            "vector":d["vector"],
        }
        for d in doc
    ]
    res=get_milvus().insert(collection_name=settings.MILVUS_COLLECTION,data=data)
    logger.info(f"已经写入{len(data)}条向量到 {settings.MILVUS_COLLECTION}")
    # 必须 flush + load：刚写入的数据还在未封存的增量段里，Milvus 不会立刻把它算进
    # query/search 的结果。少了这一步，紧接着的 count() 会返回 0、search() 一条都搜不到，
    # 表现为"首次入库明明说成功了，知识库却像是空的"。
    flush_and_load()
    return res.get("insert_count",len(data))

def search(
        query_vector:List[float],
        top_k:int=None,
        score_threshold:float=None
)->List[Dict]:
    """
    向量检索：返回按相似度降序排列的命中文档。
    每个结果形如：{text, title, source, chunk_index, score}
    score 为余弦相似度（0~1，越大越相关）。
    """
    top_k=top_k or settings.RETRIEVE_TOP_K
    score_threshold=score_threshold if score_threshold is not None else settings.RETRIEVE_SCORE_THRESHOLD

    if not collection_exists():
        logger.warning("集合不存在，返回空结果（请先运行 scripts/ingest_kb.py 入库）")
        return []
    client=get_milvus()

    try:
        results = client.search(
            collection_name=settings.MILVUS_COLLECTION,
            data=[query_vector],
            # 必须显式指定 anns_field：集合里现在有 vector（稠密）和 sparse（BM25）
            # 两个向量字段，不指定的话 Milvus 无法判断搜哪个，会直接报
            # "multiple anns_fields exist, please specify a anns_field in search_params"。
            anns_field="vector",
            limit=top_k,
            output_fields=["text", "title", "source", "chunk_index"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
        )
    except MilvusException as e:
        # 判断是不是集合未加载
        if "collection not loaded" in str(e).lower():
            logger.warning("集合未加载，执行load_collection")
            client.load_collection(settings.MILVUS_COLLECTION)
            results = client.search(
                collection_name=settings.MILVUS_COLLECTION,
                data=[query_vector],
                anns_field="vector",
                limit=top_k,
                output_fields=["text", "title", "source", "chunk_index"],
                search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
            )
        else:
            # 其他milvus异常直接抛出
            raise e


    hits=[]
    # MilvusClient.search 返回: [[{id, distance, entity:{...}}, ...]]
    for hit in results[0]:
        # 注意：集合的 metric_type 是 COSINE，Milvus 对 COSINE 返回的 distance
        # 本身就是余弦相似度（越大越相关），**不是**距离，所以这里直接取用。
        # 曾经写成 score=1.0-float(hit["distance"])——那会把相关性整个反过来：
        # 最相关的切片被阈值滤掉，留下最不相关的，且分数含义也不再是"相似度"。
        score=float(hit["distance"])

        if score < score_threshold:
            continue
        entity=hit["entity"]
        hits.append(
            {
                "text": entity["text"],
                "title": entity.get("title", ""),
                "source": entity.get("source", ""),
                "chunk_index": entity.get("chunk_index", 0),
                "score": round(score, 4),
                # 标明分数的含义：稠密分支是余弦相似度，混合检索是融合分，两者不可比
                "score_kind": "cosine",
            }
        )

    return hits


def hybrid_search(
    query_vector: List[float],
    query_text: str,
    top_k: int = None,
    candidates: int = None,
) -> List[Dict]:
    """
    混合检索：稠密向量 + BM25 词面，由 Milvus 的 ranker 融合（即重排）。

    为什么要混合：向量检索擅长"换了说法也能找到"，但对**精确词面**不敏感——
    用户问的若是原文里出现过的专有名词、型号、条款名，词面召回往往更准。
    两路各自召回再融合，比任何单一路都稳。

    参数
    ----
    query_vector : 稠密检索用的向量（调用方已经算好，避免重复 embedding）
    query_text   : 原始问句。BM25 分支直接吃文本，由 Milvus 用同一个分词器处理，
                   所以**不要**自己分词或传向量
    top_k        : 最终返回条数
    candidates   : 每路融合前各取多少候选（取大一些给重排留空间）

    返回结构与 `search()` 一致，但 `score` 是**融合分**（见 score_kind），
    与余弦相似度不可比，因此这里不做阈值过滤——语义门槛由调用方用稠密分支单独把。
    """
    if not collection_exists():
        logger.warning("集合不存在，返回空结果（请先运行 scripts/ingest_kb.py 入库）")
        return []

    top_k = top_k or settings.RETRIEVE_TOP_K
    candidates = candidates or settings.HYBRID_CANDIDATES
    strategy = (settings.RERANK_STRATEGY or "rrf").lower()

    if strategy == "weighted":
        ranker = WeightedRanker(
            settings.HYBRID_DENSE_WEIGHT, settings.HYBRID_SPARSE_WEIGHT
        )
    else:
        if strategy != "rrf":
            logger.warning("未知的 RERANK_STRATEGY=%s，回退为 rrf", strategy)
            # 记下"实际生效"的策略名：下面 score_kind 要用它，
            # 否则会出现"标着 fusion-<拼错的策略>"这种自相矛盾的分数类型
            strategy = "rrf"
        ranker = RRFRanker(k=settings.RRF_K)

    reqs = [
        AnnSearchRequest(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=candidates,
        ),
        AnnSearchRequest(
            data=[query_text],
            anns_field="sparse",
            param={"metric_type": "BM25", "params": {}},
            limit=candidates,
        ),
    ]

    client = get_milvus()
    try:
        results = client.hybrid_search(
            collection_name=settings.MILVUS_COLLECTION,
            reqs=reqs,
            ranker=ranker,
            limit=top_k,
            output_fields=["text", "title", "source", "chunk_index"],
        )
    except MilvusException as e:
        if "collection not loaded" in str(e).lower():
            logger.warning("集合未加载，执行 load_collection 后重试")
            client.load_collection(settings.MILVUS_COLLECTION)
            results = client.hybrid_search(
                collection_name=settings.MILVUS_COLLECTION,
                reqs=reqs,
                ranker=ranker,
                limit=top_k,
                output_fields=["text", "title", "source", "chunk_index"],
            )
        else:
            raise

    hits = []
    for hit in results[0]:
        entity = hit["entity"]
        hits.append(
            {
                "text": entity["text"],
                "title": entity.get("title", ""),
                "source": entity.get("source", ""),
                "chunk_index": entity.get("chunk_index", 0),
                "score": round(float(hit["distance"]), 4),
                "score_kind": f"fusion-{strategy}",
            }
        )
    return hits

def count()->int:
    """
    返回集合中的向量总数。
    用 query + count(*) 聚合（实时准确）；get_collection_stats 的行数统计
    存在最终一致性延迟，不适合刚写入后立即判断。
    """
    if not collection_exists():
        return 0
    res=get_milvus().query(
        collection_name=settings.MILVUS_COLLECTION,
        filter="id >= 0",
        output_fields=["count(*)"]
    )
    return int(res[0]["count(*)"]) if res else 0

def flush_and_load():
    if not collection_exists():
        return 0
    client=get_milvus()
    client.flush(collection_name=settings.MILVUS_COLLECTION)
    client.load_collection(collection_name=settings.MILVUS_COLLECTION)
    return "success"
