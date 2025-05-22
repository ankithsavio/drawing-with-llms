def extract_json_response(content: str):
    content = content.split("```json")[1]
    content = content.split("```")[0]
    return content.strip()
