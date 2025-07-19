
import os
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def is_image_file(file_path):
    """Check if file is a valid image"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    return os.path.splitext(file_path.lower())[1] in image_extensions

def safe_read_file(file_path):
    """Safely read file based on its type"""
    try:
        if is_image_file(file_path):
            # Handle image files with PIL
            with Image.open(file_path) as img:
                logger.info(f"Successfully opened image: {file_path} ({img.size})")
                return {"type": "image", "size": img.size, "format": img.format}
        else:
            # Handle text files
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return {"type": "text", "content": content}
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return {"type": "error", "message": str(e)}
