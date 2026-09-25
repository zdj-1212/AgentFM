"""
知识库入库入口（命令行包装）
==============================
用法（项目根目录）：
    uv run python -m scripts.ingest_kb            # 增量入库
    uv run python -m scripts.ingest_kb --reset    # 清空后全量重灌
"""
import argparse

from app.core.logger import get_logger
from app.knowledge.ingest import ingest

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentFM 知识库入库")
    parser.add_argument("--reset", action="store_true", help="清空集合后重新入库")
    args = parser.parse_args()

    count = ingest(reset=args.reset)
    if count > 0:
        logger.info("知识库入库完成，本次写入 %s 条", count)
    else:
        logger.warning("未写入任何数据，请检查 corpus 目录")


if __name__ == "__main__":
    main()
