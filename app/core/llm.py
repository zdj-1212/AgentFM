from functools import lru_cache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config.settings import settings

@lru_cache(maxsize=1)
def get_llm()->BaseChatModel:
    if settings.LLM_PROVIDER=="openai_compatible":
        model_kwargs = {}
        extra_body = {"enable_thinking":False,

                      }

        return ChatOpenAI(
            model=settings.LLM_MODEL,
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            extra_body=extra_body,
            # 必须显式设置：这两个参数默认是 None，会退化成 openai SDK 的默认值
            # （600 秒超时 + 2 次重试），一次卡住的大模型调用足以把请求挂住十几分钟。
            request_timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    raise ValueError(f"不支持的 LLM_PROVIDER: {settings.LLM_PROVIDER}")

if __name__ == '__main__':
    print(settings.LLM_BASE_URL,settings.LLM_API_KEY)