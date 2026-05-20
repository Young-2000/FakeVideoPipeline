import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Pipeline 运行日志目录；可用 .env 覆盖为绝对路径
PIPELINE_LOG_DIR = Path(
    os.getenv("PIPELINE_LOG_DIR", str(_PROJECT_ROOT / "logs"))
).expanduser()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()

MAX_COMPRESSED_VIDEO_BYTES = int(os.getenv("MAX_COMPRESSED_VIDEO_BYTES", str(18 * 1024 * 1024)))

# 检索由模型决定何时停止；此处仅为防止死循环的「安全上限」（步数）
SEARCH_SAFETY_MAX_STEPS = int(os.getenv("SEARCH_SAFETY_MAX_STEPS", "30"))
# 「handle + 关键词叠加」每轮最多几批新关键词（每批 2-3 词由模型给出）
SEARCH_KEYWORD_BATCH_MAX_ROUNDS = int(os.getenv("SEARCH_KEYWORD_BATCH_MAX_ROUNDS", "5"))

# YouTube 下载：解决 "Sign in to confirm you're not a bot"
# 方式 A（推荐）：用浏览器扩展导出 Netscape 格式 cookies.txt，填绝对路径
YT_DLP_COOKIES_FILE = os.getenv("YT_DLP_COOKIES_FILE", "").strip()
# 方式 B：从本机浏览器读已登录 YouTube 的 Cookie，格式 browser 或 browser:profile，如 chrome、safari、firefox
YT_DLP_COOKIES_FROM_BROWSER = os.getenv("YT_DLP_COOKIES_FROM_BROWSER", "").strip()

# YouTube 解签 / n challenge 需要 JS 运行时；默认可用 node（需本机已安装 node 且在 PATH）
# 多个用逗号：node,deno,bun。也可 node:/opt/homebrew/bin/node
YT_DLP_JS_RUNTIMES = os.getenv("YT_DLP_JS_RUNTIMES", "node,deno").strip()


def get_llm_client():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
