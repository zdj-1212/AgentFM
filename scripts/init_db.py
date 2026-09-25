"""
数据库初始化脚本
=================
建 MySQL 表结构 + 补列迁移 + 写入演示数据（演示账号/订单/物流）。

幂等：可以重复执行。需要特别注意的两步是：
1. 给已存在的库补上新增列（create_all 不会改已存在的表）；
2. 清理 user_id 为空的历史会话——那些是"用户体系上线前"的无主数据，无法安全归属。

用法（项目根目录）：
    uv run python -m scripts.init_db
"""
from app.core.logger import get_logger
from app.database import mysql_client

logger = get_logger(__name__)


def main() -> None:
    # init_db 内部依次完成：建库 -> 建表 -> 补列 -> 清理无主旧数据 -> 演示账号 -> 演示订单
    mysql_client.init_db()

    logger.info("数据库初始化完成")


if __name__ == "__main__":
    main()
