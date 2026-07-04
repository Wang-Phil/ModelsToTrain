# -*- coding: utf-8 -*-
"""快速测试 DeepSeek API 是否连通，带超时，避免无响应挂起。"""
import sys
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL, DEEPSEEK_REQUEST_TIMEOUT

# 测试时用较短超时（秒），可传参覆盖，如: python test_api.py 30
TIMEOUT = float(sys.argv[1]) if len(sys.argv) > 1 else min(30, DEEPSEEK_REQUEST_TIMEOUT)

def main():
    print(f"API_BASE: {DEEPSEEK_API_BASE}")
    print(f"MODEL: {DEEPSEEK_MODEL}")
    print(f"Timeout: {TIMEOUT}s")
    print("Calling API...")
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE,
            timeout=TIMEOUT,
        )
        r = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": "回复一个字：好"}],
            max_tokens=20,
        )
        print("Reply:", r.choices[0].message.content)
        print("OK")
    except Exception as e:
        print("Error:", type(e).__name__, str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
