"""
用户认证服务
=============
面向上层（Streamlit / FastAPI / CLI）提供注册、登录与令牌签发能力，封装：

1. 口令哈希：**pwdlib**（底层 argon2id，含每用户独立随机盐与算法参数自动升级）；
2. 注册校验：用户名规则、密码长度、重复注册，错误以 AuthError 抛出（消息可直接展示）；
3. 登录令牌：**PyJWT** 签发的 HS256 JWT，带过期时间；另有 token_version 实现服务端吊销。

关于第三方库（此处曾是"只用标准库"的设计，现按需求改为引入依赖）
----------------------------------------------------------------
口令与令牌这两块从标准库手写换成了成熟库，理由：

- **手写口令哈希的迭代参数是我自己拍脑袋定的**（原来 260,000 次 PBKDF2-SHA256），
  而 OWASP 当前对该算法的建议是 600,000 次——也就是说这个值是偏低的，
  且随硬件发展需要人工跟进。换 argon2id 后，防护来自内存硬度（每次哈希占 64MiB），
  不是靠堆迭代次数，而且默认参数由库维护。
- **密钥长度、算法白名单、时钟偏移这类细节**交给库比自己写可靠。
  PyJWT 会主动警告过短的 HMAC 密钥，这正是本模块原先只在文档里提醒过的事。

注意 passlib 不要选：它 2020 年后未再发布，且与 bcrypt 5.x 已不兼容
（实测哈希普通口令直接抛 ValueError）。pwdlib 正是为替代它而出现的。

迁移历史
--------
换算法的当时，库里存的是旧格式（标准库 PBKDF2）的哈希，而**哈希无法批量重算**
（没有明文），所以先做了一段时间的"登录时就地升级"：旧格式仍能验，验过后重写为 argon2id。
所有账号迁完之后，那段兼容校验逻辑已删除。

本模块因此**只认当前算法**：任何非 argon2 的哈希都验不过（见 verify_and_upgrade 的
格式诊断分支），这是刻意的——保留旧校验路径等于让一套已废弃的算法长期留在攻击面上。
"""
import re
import time
import warnings
from functools import lru_cache
from typing import Dict,Optional, Tuple

import jwt
from pwdlib import PasswordHash

from config.settings import settings
from app.core.logger import get_logger
from app.database import mysql_client

logger=get_logger(__name__)

# 口令哈希器：recommended() 当前等价于 argon2id（m=64MiB, t=3, p=4）
_hasher = PasswordHash.recommended()

# 当前唯一认可的口令哈希前缀。历史上有过一段"兼容旧 PBKDF2"的过渡期，
# 迁移完成后已删除；这里保留常量只为在碰到旧哈希时给出**可诊断**的报错，
# 而不是让用户看到一句莫名的"用户名或密码错误"。
_ARGON2_PREFIX = "$argon2"

# JWT 算法：固定 HS256，且**解码时必须显式传同样的白名单**——
# 不传白名单等于把"用哪种算法"交给令牌的持有者决定（alg=none 之类伪造的入口）。
_JWT_ALGORITHM = "HS256"

# PyJWT 对短于 32 字节的 HMAC 密钥会**每次 encode/decode 都**发一次告警，
# 而 decode 是每个请求都要走的路径，留着会把日志刷满。
# 这里关掉它，改成模块加载时用 _warn_if_weak_secret 明确提醒一次——
# 信息没有丢失，只是从"每请求一条"变成"启动一条、且说清该怎么办"。
warnings.filterwarnings("ignore", category=jwt.InsecureKeyLengthWarning)

# HMAC-SHA256 的建议密钥长度（RFC 7518 §3.2）
_MIN_SECRET_BYTES = 32

# 明确"已公开、等于没有鉴权"的密钥取值：空值，以及仓库里那串开发默认值。
_INSECURE_SECRETS = {"", "agentfm-dev-secret-change-me"}


def assert_secret_key_is_safe() -> None:
    """
    拒绝用公开的默认密钥启动。

    AUTH_SECRET_KEY 是签发登录令牌的 HMAC 密钥：一旦它还是仓库里那串公开默认值，
    **任何人都能伪造任意用户的登录令牌**，整套鉴权形同虚设。而"记得改掉"这件事靠
    文档是管不住的（README 里写了好几遍也照样会被跳过），所以做成启动即失败。

    只对"已知公开的取值"硬失败；自定义但偏短的密钥仍只警告（见 _warn_if_weak_secret）——
    那已经是使用者的主动选择，不该由框架替他决定。
    本地开发想用默认值，把 DEBUG 打开即可跳过。
    """
    secret = (settings.AUTH_SECRET_KEY or "").strip()
    if settings.DEBUG:
        return
    if secret in _INSECURE_SECRETS:
        raise RuntimeError(
            "AUTH_SECRET_KEY 仍是公开的默认值（或为空）——任何人拿到这个值都能伪造登录令牌。\n"
            "请生成一个随机密钥填进 .env：\n"
            '    python -c "import secrets;print(secrets.token_urlsafe(48))"\n'
            "仅在本地开发时可设 DEBUG=true 跳过这项检查。"
        )


def _warn_if_weak_secret() -> None:
    """检查一次令牌签名密钥的长度，过短就明确告知后果与改法。"""
    secret = (settings.AUTH_SECRET_KEY or "").encode("utf-8")
    if len(secret) < _MIN_SECRET_BYTES:
        logger.warning(
            "AUTH_SECRET_KEY 只有 %d 字节，低于 HMAC-SHA256 建议的 %d 字节。"
            "当前仍可正常签发/校验，但密钥空间偏小、更容易被离线爆破。"
            "建议在 .env 里换成随机长字符串：python -c \"import secrets;print(secrets.token_urlsafe(48))\"",
            len(secret), _MIN_SECRET_BYTES,
        )


# 在模块加载时执行，而不是等某个入口来调用：
# 只要有人用到认证能力（API / 网页端 / CLI / 冒烟测试），配置不对就当场失败，
# 不会出现"服务看着起来了、其实谁都能伪造令牌"这种最糟的情况。
assert_secret_key_is_safe()
_warn_if_weak_secret()

# 用户名：字母 / 数字 / 下划线 / 中文，2~32 个字符
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_一-龥]{2,32}$")
# 密码长度上限：防止超长口令让 PBKDF2 变成 CPU 拒绝服务
_PASSWORD_MAX_LENGTH = 128


class AuthError(Exception):
    """认证/注册失败。消息为面向用户的中文提示，调用方可直接展示。"""


# ---------------- 口令哈希 ----------------

def hash_password(password: str) -> str:
    """生成口令哈希（argon2id，盐由库内部每次随机生成，同样的口令得到不同哈希）。"""
    return _hasher.hash(password)


def verify_and_upgrade(password: str, stored: str) -> Tuple[bool, Optional[str]]:
    """
    校验口令，并在需要时返回**应当写回的新哈希**（None 表示无需升级）。

    两个用途合一，因为升级必须发生在校验通过之后——重新哈希需要明文口令，
    校验没过根本拿不到，所以"该不该升级"只能在校验的同时判断。
    交给 pwdlib 的 verify_and_update：它会在算法参数过时（比如以后提高了
    默认内存开销）时返回新哈希，由调用方写回。
    """
    if not stored:
        return False, None

    # 只做诊断，不做兼容：碰到非 argon2 的哈希说明这是"算法迁移前"遗留的数据
    # （比如从旧备份恢复的库）。这种情况下给出明确的日志，
    # 否则用户只会看到一句"用户名或密码错误"，根本无从排查。
    if not stored.startswith(_ARGON2_PREFIX):
        logger.error(
            "口令哈希不是当前算法（argon2id），该账号无法登录，需要重置密码。"
            "前缀=%r；若这是从旧备份恢复的库，请重新跑一遍迁移", stored[:14]
        )
        return False, None

    try:
        return _hasher.verify_and_update(password, stored)
    except Exception as e:
        logger.warning("口令校验失败（哈希格式无法识别）: %s", e)
        return False, None


def verify_password(password: str, stored: str) -> bool:
    """只校验口令、不关心升级的便捷入口。任何异常都返回 False，不抛给调用方。"""
    return verify_and_upgrade(password, stored)[0]


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    """
    一个固定的假哈希，用于"用户不存在"时消耗等量时间（见 authenticate）。

    必须是**当前算法**的哈希：如果这里还留旧格式，等于按用户是否存在走两条不同
    耗时的分支，那正是时序侧信道本身。
    """
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
        "role": user.role or "user",
    }


# 角色常量与判定
#
# 三种角色**职责分离**，互不包含：
#   user  普通用户：提问、可请求转人工
#   agent 坐席    ：接入待处理会话、以人工身份回复
#   admin 管理员  ：运营总览（**只读**，不能代替坐席回复）
#
# admin 刻意不是"agent 的超集"：管理员掌握的是查看与分析能力，
# 而不是冒充客服对用户说话的能力。两者合一会让一个被盗的管理员账号
# 可以直接向任意用户发消息，风险面明显更大。
ROLE_USER = "user"
ROLE_AGENT = "agent"
ROLE_ADMIN = "admin"

_VALID_ROLES = (ROLE_USER, ROLE_AGENT, ROLE_ADMIN)


def is_admin(user: Optional[Dict]) -> bool:
    """判断用户是否为管理员。缺失/未知角色一律当作普通用户（失败时收紧，而不是放宽）。"""
    return bool(user) and user.get("role") == ROLE_ADMIN


def is_agent(user: Optional[Dict]) -> bool:
    """判断用户是否为坐席。同样从严：角色缺失或未知都不算。"""
    return bool(user) and user.get("role") == ROLE_AGENT


def require_admin(user: Optional[Dict]) -> Dict:
    """
    管理员权限闸门：不是管理员就抛 PermissionError；是则原样返回 user。

    所有"能跨用户看数据"的入口都必须先过这一关。做成显式函数而不是在各处写
    `if user["role"] != "admin"`，是为了让这类检查有个统一的、可搜索的落点——
    漏写一处就等于多开一个越权口子。
    """
    if not is_admin(user):
        logger.warning("越权访问管理员接口被拒绝: user=%s", (user or {}).get("username"))
        raise PermissionError("需要管理员权限")
    return user


def require_agent(user: Optional[Dict]) -> Dict:
    """
    坐席权限闸门：不是坐席就抛 PermissionError。

    注意它**不放过管理员**——管理员进不了坐席的队列与回复入口，这是刻意的：
    运营台只读，代客回复是坐席的职责。要调整这条边界，应该改的是角色定义，
    而不是在这里给 admin 开后门。
    """
    if not is_agent(user):
        logger.warning("越权访问坐席接口被拒绝: user=%s", (user or {}).get("username"))
        raise PermissionError("需要坐席权限")
    return user


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

    valid, upgraded = verify_and_upgrade(password, user.password_hash)
    if not valid:
        logger.info("登录失败（密码错误）: %s", username)
        raise AuthError("用户名或密码错误")

    # 参数升级：pwdlib 认为存储的哈希参数已过时（例如以后提高了默认内存开销）时，
    # 会在这里返回一个新哈希，就地写回。只在登录成功时做——重新哈希需要明文口令，
    # 也只有此刻才拿得到。失败不影响登录本身（升级是优化，不是登录的必要条件）。
    if upgraded:
        try:
            mysql_client.update_password_hash(user.id, upgraded)
            logger.info("已将账号 %s 的口令哈希升级到最新参数", username)
        except Exception as e:
            logger.warning("口令哈希升级失败（不影响本次登录）: %s", e)

    logger.info("用户登录成功: %s (id=%s)", user.username, user.id)
    return _to_public(user)


def get_user(user_id: int) -> Optional[Dict]:
    """按 id 取用户公开信息；不存在返回 None（令牌验签通过后用它确认账号仍在）。"""
    user = mysql_client.get_user_by_id(user_id)
    return _to_public(user) if user else None


# ---------------- 令牌（PyJWT 签发的标准 JWT） ----------------

def create_token(user_id: int, username: str) -> str:
    """
    签发登录令牌：标准 JWT（HS256）。

    payload 里带签发那一刻的 token_version（ver），校验时会与库里的当前值比对，
    不一致就说明该令牌已被吊销（用户退出过登录，或改过密码）。

    ver 由本函数自己从库里读**当前值**，不接受调用方传入：
    令牌只能用签发那一刻的最新版本签，不存在"用旧版本签"的合法场景；
    留个参数出去，早晚会有人传了个过期的版本，签出一个刚出生就失效的令牌。

    关于 JWT 与吊销：JWT 是自包含 + 验签的，**天生无法撤销**（签发后到过期前一直有效）。
    所以这里必须保留 ver 字段配合服务端的 token_version 比对，
    不能因为"换成了标准 JWT"就以为吊销也不需要了。
    """
    user = mysql_client.get_user_by_id(user_id)
    now = int(time.time())
    payload = {
        # sub 按 JWT 规范用字符串
        "sub": str(user_id),
        "u": username,
        "ver": int((user.token_version if user else 0) or 0),
        "iat": now,
        "exp": now + settings.AUTH_TOKEN_TTL_HOURS * 3600,
    }
    return jwt.encode(payload, settings.AUTH_SECRET_KEY, algorithm=_JWT_ALGORITHM)


def parse_token(token: str) -> Dict:
    """
    校验令牌**签名与有效期**，返回 {"user_id", "username", "token_version"}；无效则抛 AuthError。

    交给 PyJWT 校验：它会验证签名、exp/nbf/iat，并且**必须显式传 algorithms 白名单**
    ——正是这个白名单挡住了 alg=none 之类的算法替换伪造。

    注意这里只做"离线"校验，**不包括吊销检查**——吊销要跟库里的 token_version 比对，
    属于 authenticate_token 的职责。单独用本函数无法发现已被吊销的令牌。
    """
    try:
        payload = jwt.decode(
            (token or "").strip(),
            settings.AUTH_SECRET_KEY,
            algorithms=[_JWT_ALGORITHM],
            # 只要 HS256 且在有效期内即可；本项目不使用 aud/iss，不开启对应校验
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("登录已过期，请重新登录")
    except jwt.InvalidTokenError as e:
        # 签名错误 / 结构损坏 / 算法不符 / 缺必需字段都落在这里，
        # 对外统一一句话，不区分细节——差异只对攻击者有意义
        logger.info("令牌校验失败: %s", type(e).__name__)
        raise AuthError("登录状态无效，请重新登录")

    return {
        "user_id": int(payload["sub"]),
        "username": str(payload.get("u", "")),
        # 老令牌没有 ver 字段，按 0 处理：它们签发于"吊销功能上线前"，
        # 等同于版本 0，会在用户第一次退出登录后自然失效。
        "token_version": int(payload.get("ver", 0)),
    }


def authenticate_token(token: str) -> Dict:
    """
    完整的令牌校验：签名 + 有效期 + **吊销状态**，返回用户公开信息；任一环节不过抛 AuthError。

    这是所有需要鉴权的入口应该调用的函数。吊销检查要读库里的 token_version，
    因此比 parse_token 多一次查询——但调用方（如 API 的 get_current_user）
    本来就要按 user_id 取用户以确认账号仍在，所以这次查询是复用的，没有额外开销。
    """
    payload = parse_token(token)

    user = mysql_client.get_user_by_id(payload["user_id"])
    if user is None:
        # 验签通过不代表账号还在（可能已被删除）
        raise AuthError("账号不存在或已被禁用")

    if payload["token_version"] != int(user.token_version or 0):
        # 版本对不上 = 该令牌在签发之后被吊销了（退出登录 / 改密码）
        raise AuthError("登录已失效，请重新登录")

    return _to_public(user)


def revoke_tokens(user_id: int) -> int:
    """
    吊销该用户**全部**已签发的令牌（退出登录 / 改密码时调用），返回新的令牌版本号。

    注意语义：这是"退出所有设备"，不是"只退出当前这一个"。
    单令牌精确吊销需要维护已吊销令牌名单（有清理与多进程一致性问题），
    对本项目这种规模不划算——现在这个方案没有额外存储、重启也不会失效。
    """
    new_version = mysql_client.bump_token_version(user_id)
    if new_version is None:
        raise AuthError("账号不存在")
    logger.info("已吊销用户 %s 的全部令牌（token_version -> %s）", user_id, new_version)
    return new_version


class AuthService:
    """认证业务服务（把上面各步骤收敛成一个可注入的对象）。"""

    hash_password = staticmethod(hash_password)
    verify_password = staticmethod(verify_password)
    register = staticmethod(register)
    authenticate = staticmethod(authenticate)
    get_user = staticmethod(get_user)
    create_token = staticmethod(create_token)
    parse_token = staticmethod(parse_token)
    authenticate_token = staticmethod(authenticate_token)
    revoke_tokens = staticmethod(revoke_tokens)
    is_admin = staticmethod(is_admin)
    is_agent = staticmethod(is_agent)
    require_admin = staticmethod(require_admin)
    require_agent = staticmethod(require_agent)


# 模块级单例，供各入口复用
auth_service = AuthService()
