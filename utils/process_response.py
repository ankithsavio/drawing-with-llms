import io

import cairosvg
from PIL import Image


def extract_json_response(content: str):
    content = content.split("```json")[1]
    content = content.split("```")[0]
    return content.strip()


def svg_to_png(svg_code: str, size: tuple = (256, 256)):
    if "viewBox" not in svg_code:
        svg_code = svg_code.replace("<svg", f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
    return Image.open(io.BytesIO(png_data)).convert("RGB").resize(size)  # type: ignore
