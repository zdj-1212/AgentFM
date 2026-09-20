"""
数据库初始化脚本
=================
建 MySQL 表结构 + 写入演示数据（订单/物流）。

用法（项目根目录）：
    uv run python -m scripts.init_db
"""
from app.core.logger import get_logger
from app.database import mysql_client

logger = get_logger(__name__)


def main() -> None:
    # 1) 按 ORM 模型建表（已存在则跳过）
    mysql_client.init_db()

    # 2) 写入演示数据（幂等）
    mysql_client.seed_demo_data()

    logger.info("✅ 数据库初始化完成")


if __name__ == "__main__":
    main()
