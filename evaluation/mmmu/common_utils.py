import os
import requests
import base64
import binascii
import hashlib
import io
from PIL import Image
from typing import List, Union

def encode_image_to_base64(image, target_size=None):
    """Encode an image to base64 string."""
    if target_size is not None:
        width, height = image.size
        # Resize the image while maintaining the aspect ratio
        if width > height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)
        image = image.resize((new_width, new_height))
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def decode_base64_to_image(base64_string):
    """Decode a base64 string to an image."""
    if isinstance(base64_string, bytes):
        s = base64_string.decode("utf-8", errors="ignore")
    else:
        s = str(base64_string)

    s = s.strip()
    if "," in s and "base64" in s[:80].lower():
        s = s.split(",", 1)[1]

    s = "".join(s.split())
    if len(s) == 0:
        raise ValueError("Empty base64 image string")

    pad = (-len(s)) % 4
    if pad:
        s = s + ("=" * pad)

    image_data = None
    last_err = None
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            image_data = decoder(s)
            break
        except (binascii.Error, ValueError) as e:
            last_err = e

    if image_data is None:
        raise ValueError(f"Invalid base64 image payload: {last_err}")

    return Image.open(io.BytesIO(image_data))

def decode_base64_to_image_file(base64_string, output_path):
    """Decode a base64 string and save it to a file."""
    image = decode_base64_to_image(base64_string)
    image.save(output_path)

def download_file(url, local_path):
    """Download a file from a URL to a local path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def md5(file_path):
    """Calculate the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def toliststr(s):
    if isinstance(s, str) and (s[0] == '[') and (s[-1] == ']'):
        return [str(x) for x in eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError