from flask import jsonify
import logging
import os
logger = logging.getLogger(__name__)

request_counts = {}

password = os.environ['password']

def authenticate_request(request):
    auth_header = request.headers.get('Authorization')

    if not auth_header:
        return False, jsonify({'error': '缺少Authorization请求头'}), 401

    try:
        auth_type, pass_word = auth_header.split(' ', 1)
    except ValueError:
        return False, jsonify({'error': 'Authorization请求头格式错误'}), 401

    if auth_type.lower() != 'bearer':
        return False, jsonify({'error': 'Authorization类型必须为Bearer'}), 401

    if pass_word != password:
        return False, jsonify({'error': '未授权'}), 401

    return True, None, None

def process_messages_for_gemini(messages, use_system_prompt=False):
    gemini_history = []
    errors = []
    system_instruction_text = ""
    is_system_phase = use_system_prompt
    for i, message in enumerate(messages):
        role = message.get('role')
        content = message.get('content')

        if isinstance(content, str):
            if is_system_phase and role == 'system':
                if system_instruction_text:
                    system_instruction_text += "\n" + content
                else:
                    system_instruction_text = content
            else:
                is_system_phase = False

                if role in ['user', 'system']:
                    role_to_use = 'user'
                elif role == 'assistant':
                    role_to_use = 'model'
                else:
                    errors.append(f"Invalid role: {role}")
                    continue

                if gemini_history and gemini_history[-1]['role'] == role_to_use:
                    gemini_history[-1]['parts'].append({"text": content})
                else:
                    gemini_history.append({"role": role_to_use, "parts": [{"text": content}]})

        elif isinstance(content, list):
            parts = []
            for item in content:
                if item.get('type') == 'text':
                    parts.append({"text": item.get('text')})
                elif item.get('type') == 'image_url':
                    image_data = item.get('image_url', {}).get('url', '')
                    if image_data.startswith('data:image/'):
                        try:
                            mime_type, base64_data = image_data.split(';')[0].split(':')[1], image_data.split(',')[1]
                            parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": base64_data
                                }
                            })
                        except (IndexError, ValueError):
                            errors.append(f"Invalid data URI for image: {image_data}")
                    else:
                        errors.append(f"Invalid image URL format for item: {item}")
                elif item.get('type') == 'file_url':
                    file_data = item.get('file_url', {}).get('url', '')
                    if file_data.startswith('data:'):
                        try:
                            mime_type, base64_data = file_data.split(';')[0].split(':')[1], file_data.split(',')[1]
                            parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": base64_data
                                }
                            })
                        except (IndexError, ValueError):
                            errors.append(f"Invalid data URI for file: {file_data}")
                    else:
                        errors.append(f"Invalid file URL format for item: {item}")

            if parts:
                if role in ['user', 'system']:
                    role_to_use = 'user'
                elif role == 'assistant':
                    role_to_use = 'model'
                else:
                    errors.append(f"Invalid role: {role}")
                    continue
                if gemini_history and gemini_history[-1]['role'] == role_to_use:
                    gemini_history[-1]['parts'].extend(parts) 
                else:
                    gemini_history.append({"role": role_to_use, "parts": parts})

    if errors:
        return gemini_history, {"parts": [{"text": system_instruction_text}]}, (jsonify({'error': errors}), 400)
    else:
        return gemini_history, {"parts": [{"text": system_instruction_text}]}, None