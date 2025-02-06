from flask import Flask, request, jsonify, Response, stream_with_context, render_template_string
import json
import os
import re
import logging
import func
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import time
import requests
from collections import deque
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any

app = Flask(__name__)

os.environ['TZ'] = 'Asia/Shanghai'

app = Flask(__name__)

app.secret_key = os.urandom(24)

formatter = logging.Formatter('%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

MAX_RETRIES = int(os.environ.get('MaxRetries', '3').strip() or '3')
MAX_REQUESTS = int(os.environ.get('MaxRequests', '2').strip() or '2')
LIMIT_WINDOW = int(os.environ.get('LimitWindow', '60').strip() or '60')

RETRY_DELAY = 1
MAX_RETRY_DELAY = 16

request_counts = {}

api_key_blacklist = set()
api_key_blacklist_duration = 60


safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": 'HARM_CATEGORY_CIVIC_INTEGRITY',
        "threshold": 'BLOCK_NONE'
    }
]
safety_settings_g2 = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "OFF"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "OFF"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "OFF"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "OFF"
    },
    {
        "category": 'HARM_CATEGORY_CIVIC_INTEGRITY',
        "threshold": 'OFF'
    }
]
@dataclass
class GeneratedText:
    text: str
    finish_reason: Optional[str] = None


class ResponseWrapper:
    def __init__(self, data: Dict[Any, Any]):
        self._data = data
        self._text = self._extract_text()
        self._finish_reason = self._extract_finish_reason()
        self._prompt_token_count = self._extract_prompt_token_count()
        self._candidates_token_count = self._extract_candidates_token_count()
        self._total_token_count = self._extract_total_token_count()
        self._thoughts = self._extract_thoughts()
        self._json_dumps = json.dumps(self._data, indent=4, ensure_ascii=False)

    def _extract_thoughts(self) -> Optional[str]:
        try:
            for part in self._data['candidates'][0]['content']['parts']:
                if 'thought' in part:
                    return part['text']
            return ""
        except (KeyError, IndexError):
            return ""

    def _extract_text(self) -> str:
        try:
            for part in self._data['candidates'][0]['content']['parts']:
                if 'thought' not in part:
                    return part['text']
            return ""
        except (KeyError, IndexError):
            return ""

    def _extract_finish_reason(self) -> Optional[str]:
        try:
            return self._data['candidates'][0].get('finishReason')
        except (KeyError, IndexError):
            return None

    def _extract_prompt_token_count(self) -> Optional[int]:
        try:
            return self._data['usageMetadata'].get('promptTokenCount')
        except (KeyError):
            return None

    def _extract_candidates_token_count(self) -> Optional[int]:
        try:
            return self._data['usageMetadata'].get('candidatesTokenCount')
        except (KeyError):
            return None

    def _extract_total_token_count(self) -> Optional[int]:
        try:
            return self._data['usageMetadata'].get('totalTokenCount')
        except (KeyError):
            return None

    @property
    def text(self) -> str:
        return self._text

    @property
    def finish_reason(self) -> Optional[str]:
        return self._finish_reason

    @property
    def prompt_token_count(self) -> Optional[int]:
        return self._prompt_token_count

    @property
    def candidates_token_count(self) -> Optional[int]:
        return self._candidates_token_count

    @property
    def total_token_count(self) -> Optional[int]:
        return self._total_token_count

    @property
    def thoughts(self) -> Optional[str]:
        return self._thoughts

    @property
    def json_dumps(self) -> str:
        return self._json_dumps

class APIKeyManager:
    def __init__(self):
        self.api_keys = re.findall(r"AIzaSy[a-zA-Z0-9_-]{33}", os.environ.get('KeyArray'))
        self.current_index = random.randint(0, len(self.api_keys) - 1)

    def get_available_key(self):
        num_keys = len(self.api_keys)
        for _ in range(num_keys):
            if self.current_index >= num_keys:
                self.current_index = 0
            current_key = self.api_keys[self.current_index]
            self.current_index += 1

            if current_key not in api_key_blacklist:
                return current_key

        logger.error("所有API key都已耗尽或被暂时禁用，请重新配置或稍后重试")
        return None

    def show_all_keys(self):
        logger.info(f"当前可用API key个数: {len(self.api_keys)} ")
        for i, api_key in enumerate(self.api_keys):
            logger.info(f"API Key{i}: {api_key[:8]}...{api_key[-3:]}")

    def blacklist_key(self, key):
        logger.warning(f"{key[:8]} → 暂时禁用 {api_key_blacklist_duration} 秒")
        api_key_blacklist.add(key)

        scheduler.add_job(lambda: api_key_blacklist.discard(key), 'date', run_date=datetime.now() + timedelta(seconds=api_key_blacklist_duration))

key_manager = APIKeyManager()
key_manager.show_all_keys()
current_api_key = key_manager.get_available_key()

def switch_api_key():
    global current_api_key
    key = key_manager.get_available_key()
    if key:
      current_api_key = key
      logger.info(f"API key 替换为 → {current_api_key[:8]}...{current_api_key[-3:]}")
    else:
      logger.error("API key 替换失败，所有API key都已耗尽或被暂时禁用，请重新配置或稍后重试")

logger.info(f"当前 API key: {current_api_key[:8]}...{current_api_key[-3:]}")

GEMINI_MODELS = [
    {"id": "text-embedding-004"},
    {"id": "gemini-1.5-flash-8b-latest"},
    {"id": "gemini-1.5-flash-8b-exp-0924"},
    {"id": "gemini-1.5-flash-latest"},
    {"id": "gemini-1.5-flash-exp-0827"},
    {"id": "gemini-1.5-pro-latest"},
    {"id": "gemini-1.5-pro-exp-0827"},
    {"id": "learnlm-1.5-pro-experimental"},
    {"id": "gemini-exp-1114"},
    {"id": "gemini-exp-1121"},
    {"id": "gemini-exp-1206"},
    {"id": "gemini-2.0-flash-exp"},
    {"id": "gemini-2.0-flash-thinking-exp-1219"},
    {"id": "gemini-2.0-flash-thinking-exp-01-21"},
    {"id": "gemini-2.0-flash"},
    {"id": "gemini-2.0-pro-exp-02-05"}
]


def is_within_rate_limit(api_key):
    now = datetime.now()
    if api_key not in request_counts:
        request_counts[api_key] = deque()

    while request_counts[api_key] and request_counts[api_key][0] < now - timedelta(seconds=LIMIT_WINDOW):
        request_counts[api_key].popleft()

    if len(request_counts[api_key]) >= MAX_REQUESTS:
        earliest_request_time = request_counts[api_key][0]
        wait_time = (earliest_request_time + timedelta(seconds=LIMIT_WINDOW)) - now
        return False, wait_time.total_seconds()
    else:
        return True, 0

def increment_request_count(api_key):
    now = datetime.now()
    if api_key not in request_counts:
        request_counts[api_key] = deque()
    request_counts[api_key].append(now)

def handle_api_error(error, attempt, current_api_key):
    if attempt > MAX_RETRIES:
        logger.error(f"{MAX_RETRIES} 次尝试后仍然失败，请修改预设或输入")
        return 0, jsonify({
            'error': {
                'message': f"{MAX_RETRIES} 次尝试后仍然失败，请修改预设或输入",
                'type': 'max_retries_exceeded'
            }
        })

    if isinstance(error, requests.exceptions.HTTPError):
        status_code = error.response.status_code

        if status_code == 400:

            try:
                error_data = error.response.json()
                if 'error' in error_data:
                    if error_data['error'].get('code') == "invalid_argument":
                        logger.error(f"{current_api_key[:8]} ... {current_api_key[-3:]} → 无效，可能已过期或被删除")
                        key_manager.blacklist_key(current_api_key)
                        switch_api_key()
                        return 0, None
                    error_message = error_data['error'].get('message', 'Bad Request')
                    error_type = error_data['error'].get('type', 'invalid_request_error')
                    logger.warning(f"400 错误请求: {error_message}")
                    return 2, jsonify({'error': {'message': error_message, 'type': error_type}})
            except ValueError:
                logger.warning("400 错误请求：响应不是有效的JSON格式")
                return 2, jsonify({'error': {'message': '', 'type': 'invalid_request_error'}})

        elif status_code == 429:
            logger.warning(
                f"{current_api_key[:8]} ... {current_api_key[-3:]} → 429 官方资源耗尽 → 立即重试..."
            )
            key_manager.blacklist_key(current_api_key)
            switch_api_key()
            return 0, None

        elif status_code == 403:
            logger.error(
                f"{current_api_key[:8]} ... {current_api_key[-3:]} → 403 权限被拒绝，该 API KEY 可能已经被官方封禁"
            )
            key_manager.blacklist_key(current_api_key)
            switch_api_key()
            return 0, None

        elif status_code == 500:
            logger.warning(
                f"{current_api_key[:8]} ... {current_api_key[-3:]} → 500 服务器内部错误 → 立即重试..."
            )
            switch_api_key()
            return 0, None

        elif status_code == 503:
            logger.warning(
                f"{current_api_key[:8]} ... {current_api_key[-3:]} → 503 服务不可用 → 立即重试..."
            )
            switch_api_key()
            return 0, None

        else:
            logger.warning(
                f"{current_api_key[:8]} ... {current_api_key[-3:]} → {status_code} 未知错误/模型不可用 → 不重试..."
            )
            switch_api_key()
            return 2, None

    elif isinstance(error, requests.exceptions.ConnectionError):
        delay = min(RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
        logger.warning(f"连接错误 → 立即重试...")
        time.sleep(delay)
        return 0, None

    elif isinstance(error, requests.exceptions.Timeout):
        delay = min(RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
        logger.warning(f"请求超时 → 立即重试...")
        time.sleep(delay)
        return 0, None

    else:
        logger.error(f"发生未知错误: {error}")
        return 0, jsonify({
            'error': {
                'message': f"发生未知错误: {error}",
                'type': 'unknown_error'
            }
        })
@app.route('/')
def index():
    main_content = "Moonfanz gemini-rProxy v2.3.5 2025-01-25"
    html_template = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script>
function copyToClipboard(text) {
  var textarea = document.createElement("textarea");
  textarea.textContent = text;
  textarea.style.position = "fixed";
  document.body.appendChild(textarea);
  textarea.select();
  try {
    return document.execCommand("copy");
  } catch (ex) {
    console.warn("Copy to clipboard failed.", ex);
    return false;
  } finally {
    document.body.removeChild(textarea);
  }
}
function copyLink(event) {
  event.preventDefault();
  const url = new URL(window.location.href);
  const link = url.protocol + '//' + url.host + '/hf/v1';
  copyToClipboard(link);
  alert('链接已复制: ' + link);
}
</script>
</head>
<body>
{{ main_content }}<br/><br/>完全开源、免费且禁止商用<br/><br/>点击复制反向代理: <a href="v1" onclick="copyLink(event)">Copy Link</a><br/>聊天来源选择"自定义(兼容 OpenAI)"<br/>将复制的网址填入到自定义端点<br/>将设置password填入自定义API秘钥<br/><br/><br/>
</body>
</html>
    """
    return render_template_string(html_template, main_content=main_content)

@app.route('/hf/v1/chat/completions', methods=['POST'])
def chat_completions():
    is_authenticated, auth_error, status_code = func.authenticate_request(request)
    if not is_authenticated:
        return auth_error if auth_error else jsonify({'error': '未授权'}), status_code if status_code else 401

    request_data = request.get_json()
    messages = request_data.get('messages', [])
    model = request_data.get('model', 'gemini-2.0-flash-exp')
    temperature = request_data.get('temperature', 1)
    max_tokens = request_data.get('max_tokens', 8192)
    show_thoughts = request_data.get('show_thoughts', False)
    stream = request_data.get('stream', False)
    use_system_prompt = request_data.get('use_system_prompt', False)
    hint = "流式" if stream else "非流"
    logger.info(f"\n{model} [{hint}] → {current_api_key[:8]}...{current_api_key[-3:]}")
    is_thinking = 'thinking' in model
    api_version = 'v1alpha' if is_thinking else 'v1beta'
    response_type = 'streamGenerateContent' if stream else 'generateContent'
    is_SSE = '&alt=sse' if stream else ''

    contents, system_instruction, error_response = func.process_messages_for_gemini(messages, use_system_prompt)

    if error_response:
        logger.error(f"处理输入消息时出错↙\n {error_response}")
        return jsonify(error_response), 400

    def do_request(current_api_key, attempt):
        isok, time_remaining = is_within_rate_limit(current_api_key)
        if not isok:
            logger.warning(f"暂时超过限额，该API key将在 {time_remaining} 秒后启用...")
            switch_api_key()
            return 0, None

        increment_request_count(current_api_key)


        url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:{response_type}?key={current_api_key}{is_SSE}"
        headers = {
            "Content-Type": "application/json",
        }

        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
            "safetySettings": safety_settings_g2 if 'gemini-2.0-flash-exp' in model else safety_settings,
        }
        if system_instruction:
            data["system_instruction"] = system_instruction

        try:
            response = requests.post(url, headers=headers, json=data, stream=True)
            response.raise_for_status()

            if stream:
                return 1, response
            else:
                return 1, ResponseWrapper(response.json())
        except requests.exceptions.RequestException as e:
            return handle_api_error(e, attempt, current_api_key)

    def generate_stream(response):
        logger.info(f"流式开始 →")
        buffer = b""
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    if line.startswith(b'data: '):
                        line = line[6:]

                    buffer += line

                    try:
                        data = json.loads(buffer.decode('utf-8'))
                        buffer = b""
                        if 'candidates' in data and data['candidates']:
                            candidate = data['candidates'][0]
                            if 'content' in candidate:
                                content = candidate['content']
                                if 'parts' in content and content['parts']:
                                    parts = content['parts']
                                    if is_thinking and not show_thoughts:
                                        parts = [part for part in parts if not part.get('thought')]
                                    if parts:
                                        text = parts[0].get('text', '')
                                        finish_reason = candidate.get('finishReason')

                                        if text:
                                            data = {
                                                'choices': [{
                                                    'delta': {
                                                        'content': text
                                                    },
                                                    'finish_reason': finish_reason,
                                                    'index': 0
                                                }],
                                                'object': 'chat.completion.chunk'
                                            }
                                            yield f"data: {json.dumps(data)}\n\n"

                            if candidate.get("finishReason") and candidate.get("finishReason") != "STOP":
                                error_message = {
                                    "error": {
                                        "code": "content_filter",
                                        "message": f"模型的响应因违反内容政策而被标记：{candidate.get('finishReason')}",
                                        "status": candidate.get("finishReason"),
                                        "details": []
                                    }
                                }
                                logger.warning(f"模型的响应因违反内容政策而被标记: {candidate.get('finishReason')}")
                                yield f"data: {json.dumps(error_message)}\n\n"
                                break

                            if 'safetyRatings' in candidate:
                                for rating in candidate['safetyRatings']:
                                    if rating['probability'] == 'HIGH':
                                        error_message = {
                                            "error": {
                                                "code": "content_filter",
                                                "message": f"模型的响应因高概率被标记为 {rating['category']}",
                                                "status": "SAFETY_RATING_HIGH",
                                                "details": [rating]
                                            }
                                        }
                                        logger.warning(f"模型的响应因高概率被标记为 {rating['category']}")
                                        yield f"data: {json.dumps(error_message)}\n\n"
                                        break
                                else:
                                    continue
                                break

                    except json.JSONDecodeError:
                        logger.debug(f"JSON解析错误, 当前缓冲区内容: {buffer}")
                        continue

                except Exception as e:
                    logger.error(f"流式处理期间发生错误: {e}, 原始数据行↙\n{line}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            else:
                yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop', 'index': 0}]})}\n\n"
                logger.info(f"流式结束 ←")
                logger.info(f"200!")
        except Exception as e:
            logger.error(f"流式处理错误↙\n{e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    attempt = 0
    success = 0
    response = None
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"第 {attempt}/{MAX_RETRIES} 次尝试 ...")
        success, response = do_request(current_api_key, attempt)

        if success == 0:
            continue
        elif success == 1 and response is None:
            continue
        elif success == 1 and stream:
            return Response(
                stream_with_context(generate_stream(response)),
                mimetype='text/event-stream'
            )
        elif success == 1 and isinstance(response, ResponseWrapper):
            try:
                text_content = response.text
                prompt_tokens = response.prompt_token_count
                completion_tokens = response.candidates_token_count
                total_tokens = response.total_token_count
                finish_reason = response.finish_reason

                if text_content == '':
                    error_message = None
                    if response._data and 'error' in response._data:
                        error_message = response._data['error'].get('message')
                    if error_message:
                        logger.error(f"生成内容失败，API 返回错误: {error_message}")
                    else:
                        logger.error(f"生成内容失败: text_content 为空")
                    continue

                if is_thinking and show_thoughts:
                    text_content = response.thoughts + '\n' + text_content

            except AttributeError as e:
                logger.error(f"处理响应失败，缺少必要的属性: {e}")
                logger.error(f"原始响应: {response._data}")
                continue

            except Exception as e:
                logger.error(f"处理响应失败: {e}")
                continue

            response_data = {
                'id': 'chatcmpl-xxxxxxxxxxxx',
                'object': 'chat.completion',
                'created': int(datetime.now().timestamp()),
                'model': model,
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': text_content
                    },
                    'finish_reason': finish_reason
                }],
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                }
            }
            logger.info(f"200!")
            return jsonify(response_data)
        elif success == 1 and isinstance(response, tuple):
            return response[1], response[0]
        elif success == 2:
            logger.error(f"{model} 可能暂时不可用，请更换模型或未来一段时间再试")
            response = {
                'error': {
                    'message': f'{model} 可能暂时不可用，请更换模型或未来一段时间再试',
                    'type': 'internal_server_error'
                }
            }
            return jsonify(response), 503
    else:
        logger.error(f"{MAX_RETRIES} 次尝试均失败，请重试或等待官方恢复")
        response = {
            'error': {
                'message': f'{MAX_RETRIES} 次尝试均失败，请重试或等待官方恢复',
                'type': 'internal_server_error'
            }
        }
        return jsonify(response), 500 if response is not None else 503

@app.route('/hf/v1/models', methods=['GET'])
def list_models():
    response = {"object": "list", "data": GEMINI_MODELS}
    return jsonify(response)

@app.route('/hf/v1/embeddings', methods=['POST'])
def embeddings():
    data = request.get_json()
    model_input = data.get("input")
    model = data.get("model", "text-embedding-004")
    if not model_input:
        return jsonify({"error": "没有提供输入"}), 400

    if isinstance(model_input, str):
        model_input = [model_input]

    gemini_request = {
        "model": f"models/{model}",
        "content": {
            "parts": [{"text": text} for text in model_input]
        }
    }

    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={current_api_key}"
    headers = {"Content-Type": "application/json"}
    try:
        gemini_response = requests.post(gemini_url, json=gemini_request, headers=headers)
        gemini_response.raise_for_status()

        response_json = gemini_response.json()
        embeddings_data = []
        if 'embedding' in response_json:
          embeddings_data.append({
              "object": "embedding",
              "embedding": response_json['embedding']['values'],
              "index": 0,
          })
        elif 'embeddings' in response_json:
          for i, embedding in enumerate(response_json['embeddings']):
              embeddings_data.append({
                  "object": "embedding",
                  "embedding": embedding['values'],
                  "index": i,
              })

        client_response = {
            "object": "list",
            "data": embeddings_data,
            "model": model,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }
        switch_api_key()
        return jsonify(client_response)

    except requests.exceptions.RequestException as e:
        print(f"请求Embeddings失败↙\: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    scheduler = BackgroundScheduler()

    scheduler.start()
    logger.info(f"Reminiproxy v2.3.5 启动")
    logger.info(f"最大尝试次数/MaxRetries: {MAX_RETRIES}")
    logger.info(f"最大请求次数/MaxRequests: {MAX_REQUESTS}")
    logger.info(f"请求限额窗口/LimitWindow: {LIMIT_WINDOW} 秒")

    app.run(debug=True, host='0.0.0.0', port=7860)