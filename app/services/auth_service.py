"""
用户认证服务
=============
面向上层（Streamlit / FastAPI / CLI）提供注册、登录与令牌签发能力，封装：

1. 口令哈希：PBKDF2-HMAC-SHA256（标准库 hashlib），每个用户独立随机盐，
   存储格式 `pbkdf2_sha256$迭代次数$盐$摘要`，校验用 hmac.compare_digest 防时序攻击；
2. 注册校验：用户名规则、密码长度、重复注册，错误以 AuthError 抛出（消息可直接展示）；
3. 登录令牌：HMAC-SHA256 签名的自包含令牌（base64url(payload).base64url(签名)），
   带过期时间，服务端无状态、无需查库即可验签。

为什么不用 passlib / bcrypt / pyjwt：
本项目的口令与令牌需求用标准库即可完整覆盖，不额外引入第三方依赖，
也就没有版本冲突与供应链面的增加。若后续要做密码轮换/更复杂鉴权再引入不迟。
"""
import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from functools import lru_cache
from typing import Dict,Optional

from config.settings import settings
from app.core.logger import get_logger
from app.database import mysql_client

logger=get_logger(__name__)

# PBKDF2 参数：迭代次数越高越抗暴力破解，代价是登录慢一点。
# 26 万次是当前较为主流的推荐量级，现代 CPU 上约几十毫秒。
PBKDF2_ITERATIONS = 260_000
_HASH_ALGO = "pbkdf2_sha256"

# 用户名：字母 / 数字 / 下划线 / 中文，2~32 个字符
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_一-龥]{2,32}$")
# 密码长度上限：防止超长口令让 PBKDF2 变成 CPU 拒绝服务
_PASSWORD_MAX_LENGTH = 128


class AuthError(Exception):
    """认证/注册失败。消息为面向用户的中文提示，调用方可直接展示。"""


# ---------------- 口令哈希 ----------------

def hash_password(password: str) -> str:
    """生成口令哈希（每次调用都用新的随机盐，同样的密码得到的哈希不同）。"""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"{_HASH_ALGO}${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """校验口令是否匹配存储的哈希。格式非法时返回 False，绝不抛异常。"""
    try:
        algo, iterations, salt_hex, digest_hex = stored.split("$")
        if algo != _HASH_ALGO:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        # 常量时间比较，避免通过响应时间逐字节试探摘要
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, AttributeError, TypeError):
        logger.warning("口令哈希格式非法，拒绝校验")
        return False


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    """一个固定的假哈希，用于"用户不存在"时消耗等量时间（见 authenticate）。"""
    return hash_password("timing-attack-placeholder")


# ---------------- 校验 ----------------

def _validate_username(username: str) -> str:
    """校验并规范化用户名，非法则抛 AuthError。"""
    username = (username or "").strip()
    if not _USERNAME_RE.match(username):
        raise AuthError("用户名需为 2~32 位的中文、字母、数字或下划线")
    return username


def _validate_password(password: str) -> str:
    """校验密码强度，非法则抛 AuthError。"""
    password = password or ""
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        raise AuthError(f"密码至少需要 {settings.PASSWORD_MIN_LENGTH} 位")
    if len(password) > _PASSWORD_MAX_LENGTH:
        raise AuthError(f"密码不能超过 {_PASSWORD_MAX_LENGTH} 位")
    return password


def _to_public(user) -> Dict:
    """把 ORM User 转成对外安全的字典（绝不包含 password_hash）。"""
    return {
        "user_id": user.id,
        "username": user.username,
        "display_name": user.display_name or user.username,
    }


# ---------------- 注册 / 登录 ----------------

def register(username: str, password: str, display_name: Optional[str] = None) -> Dict:
    """
    注册新用户，成功返回 {"user_id", "username", "display_name"}。
    用户名非法 / 密码太弱 / 用户名已被占用 都会抛 AuthError。
    """
    username = _validate_username(username)
    password = _validate_password(password)

    if mysql_client.get_user_by_username(username) is not None:
        raise AuthError(f"用户名「{username}」已被注册，请换一个")

    user = mysql_client.create_user(
        username=username,
        password_hash=hash_password(password),
        display_name=(display_name or "").strip() or username,
    )
    logger.info("新用户注册成功: %s (id=%s)", user.username, user.id)
    return _to_public(user)


def authenticate(username: str, password: str) -> Dict:
    """
    校验用户名密码，成功返回用户信息字典，失败抛 AuthError。

    失败一律返回同一句"用户名或密码错误"，不区分"用户不存在"和"密码错误"，
    否则等于给攻击者提供了一个免费的用户名枚举接口。
    """
    username = (username or "").strip()
    password = password or ""
    user = mysql_client.get_user_by_username(username) if username else None

    if user is None:
        # 用户不存在时也做一次等价耗时的哈希校验，抹平"存在/不存在"的响应时间差
        verify_password(password, _dummy_hash())
        raise AuthError("用户名或密码错误")

    if not verify_password(password, user.password_hash):
        logger.info("登录失败（密码错误）: %s", username)
        raise AuthError("用户名或密码错误")

    logger.info("用户登录成功: %s (id=%s)", user.username, user.id)
    return _to_public(user)


def get_user(user_id: int) -> Optional[Dict]:
    """按 id 取用户公开信息；不存在返回 None（令牌验签通过后用它确认账号仍在）。"""
    user = mysql_client.get_user_by_id(user_id)
    return _to_public(user) if user else None


# ---------------- 令牌 ----------------

def _b64e(raw: bytes) -> str:
    """base64url 编码并去掉 '=' 填充（令牌里不需要，也避免 URL 转义问题）。"""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    """base64url 解码，自动补齐被去掉的 '=' 填充。"""
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload_b64: str) -> str:
    """对 payload 做 HMAC-SHA256 签名。密钥泄露 = 任何人都能伪造任意用户。"""
    mac = hmac.new(
        settings.AUTH_SECRET_KEY.encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    )
    return _b64e(mac.digest())


def create_token(user_id: int, username: str) -> str:
    """签发登录令牌：base64url(payload) + '.' + base64url(签名)。"""
    payload = {
        "uid": user_id,
        "u": username,
        "exp": int(time.time()) + settings.AUTH_TOKEN_TTL_HOURS * 3600,
    }
    payload_b64 = _b64e(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return f"{payload_b64}.{_sign(payload_b64)}"


def parse_token(token: str) -> Dict:
    """
    校验令牌签名与有效期，返回 {"user_id", "username"}；无效则抛 AuthError。

    先验签再解析内容：签名不对就完全不信任 payload 里的任何字节。
    """
    try:
        payload_b64, signature = (token or "").strip().split(".")
    except ValueError:
        raise AuthError("登录状态无效，请重新登录")

    if not hmac.compare_digest(_sign(payload_b64), signature):
        raise AuthError("登录状态无效，请重新登录")

    try:
        payload = json.loads(_b64d(payload_b64))
    except (ValueError, TypeError):
        raise AuthError("登录状态无效，请重新登录")

    if int(payload.get("exp", 0)) < int(time.time()):
        raise AuthError("登录已过期，请重新登录")

    return {"user_id": int(payload["uid"]), "username": str(payload["u"])}


class AuthService:
    """认证业务服务（把上面各步骤收敛成一个可注入的对象）。"""

    hash_password = staticmethod(hash_password)
    verify_password = staticmethod(verify_password)
    register = staticmethod(register)
    authenticate = staticmethod(authenticate)
    get_user = staticmethod(get_user)
    create_token = staticmethod(create_token)
    parse_token = staticmethod(parse_token)


# 模块级单例，供各入口复用
auth_service = AuthService()
