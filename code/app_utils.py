import base64
import random
import cv2
import numpy as np


def img_to_base64(filepath):
    with open(filepath, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
def rand(num_bits=30):
    print("Random input bit message")
    random_str = ''.join([str(random.randint(0, 1)) for _ in range(num_bits)])
    return random_str

def calculate_similarity_percentage(str1, str2):
    if len(str1) == 0:
        return "Original copyright watermark unknown"
    elif len(str1) != len(str2):
        return "Input and output watermark lengths differ"
    total_length = len(str1)
    same_count = sum(1 for x, y in zip(str1, str2) if x == y)
    similarity_percentage = (same_count / total_length) * 100
    return f"{similarity_percentage}%"

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR NumPy image to RGB using OpenCV.
    
    Args:
        img_bgr (np.ndarray): Input image in BGR format (H,W,3)
    
    Returns:
        np.ndarray: Output image in RGB format (H,W,3), same dtype as input
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB NumPy image to BGR using OpenCV.

    Args:
        img_rgb (np.ndarray): Input image in RGB format (H,W,3)

    Returns:
        np.ndarray: Output image in BGR format (H,W,3), same dtype as input
    """
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def encode_ascii(text: str, errors: str = "strict") -> str:
    """
    Encode text to ASCII, then output a '0'/'1' bitstring (MSB-first, 8 bits/byte).
    errors: 'strict' | 'ignore' | 'replace'
    """
    b = text.encode("ascii", errors=errors)  # raises if non-ASCII and errors='strict'
    return ''.join(f'{byte:08b}' for byte in b)


def decode_ascii(bits: str, errors: str = "strict") -> str:
    """
    Decode a '0'/'1' bitstring (length must be multiple of 8) back to ASCII text.
    errors: passed to .decode() for safety, though ASCII should be exact.
    """
    if any(c not in '01' for c in bits):
        raise ValueError("bits must contain only '0' and '1'")
    if len(bits) % 8 != 0:
        raise ValueError("bitstring length must be a multiple of 8")

    byts = bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))
    return byts.decode("ascii", errors=errors)

def split_into_tiles_128(img: np.ndarray, pad_mode: str = "edge"):
    """
    Split HxWxC (or HxW) image into 128x128 tiles.
    - Pads bottom/right so all tiles are exactly 128x128.
    - Returns: tiles (list of np.ndarray), coords (list of (y,x)),
               orig_hw ((H,W)), padded_hw ((Hp,Wp))
    """
    if img is None:
        raise ValueError("split_into_tiles_128: img is None")

    # Accept grayscale HxW → HxWx1
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC or HxW, got shape {img.shape}")

    H, W, C = img.shape
    tile = 128

    pad_h = (tile - (H % tile)) % tile
    pad_w = (tile - (W % tile)) % tile

    # Pad on bottom/right only
    img_padded = np.pad(
        img,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode=pad_mode  # 'edge'/'reflect'/'constant'
    )

    Hp, Wp, _ = img_padded.shape
    tiles, coords = [], []

    for y in range(0, Hp, tile):
        for x in range(0, Wp, tile):
            patch = img_padded[y:y+tile, x:x+tile, :]
            # make contiguous to avoid stride issues later
            tiles.append(np.ascontiguousarray(patch))
            coords.append((y, x))

    return tiles, coords, (H, W), (Hp, Wp)

def combine_tiles_ordered(list_container, num_child_on_width_size, num_child_on_height_size):
    """
    Combine HWC (or HW) tiles into a single HWC image, assuming row-major order.
    list_container: [tile0, tile1, ...] where tile shapes are (H,W,C) or (H,W)
    num_child_on_width_size: number of tiles per row (columns)
    num_child_on_height_size: number of rows
    """
    if not list_container:
        raise ValueError("list_container is empty.")

    num_needed = num_child_on_width_size * num_child_on_height_size
    if len(list_container) < num_needed:
        raise ValueError(f"Need {num_needed} tiles, got {len(list_container)}.")

    tiles = list_container[:num_needed]

    # Normalize to HWC
    t0 = tiles[0]
    if t0.ndim == 2:
        H, W = t0.shape
        C = 1
    elif t0.ndim == 3:
        H, W, C = t0.shape
    else:
        raise ValueError(f"Tile must be HW or HWC, got shape {t0.shape}")

    dtype = t0.dtype
    norm_tiles = []
    for i, t in enumerate(tiles):
        if t.ndim == 2:
            t = t[..., None]
        if t.shape != (H, W, C):
            raise ValueError(f"Tile {i} shape {t.shape} != {(H, W, C)}")
        if t.dtype != dtype:
            raise ValueError(f"Tile {i} dtype {t.dtype} != {dtype}")
        norm_tiles.append(np.ascontiguousarray(t))

    out = np.empty((num_child_on_height_size * H, num_child_on_width_size * W, C), dtype=dtype)

    for idx, tile in enumerate(norm_tiles):
        r = idx // num_child_on_width_size
        c = idx % num_child_on_width_size
        out[r*H:(r+1)*H, c*W:(c+1)*W, :] = tile

    return out  # if grayscale and you want HW: out[..., 0]

def split_bits_30(bitstr: str):
    """
    bitstr: chuỗi '0'/'1' (ví dụ: bit_input = sha256_bitstring(text_input))
    return: (chunks: list[str] mỗi chuỗi dài 30, last_real_len: int)
    """
    if not isinstance(bitstr, str):
        raise TypeError("bitstr must be a string of '0'/'1'")
    if any(c not in "01" for c in bitstr):
        raise ValueError("bitstr must contain only '0' and '1'")

    chunks = [bitstr[i:i+30] for i in range(0, len(bitstr), 30)]
    last_real_len = len(chunks[-1]) if chunks else 0
    if last_real_len == 0:
        return [], 0
    if last_real_len < 30:
        chunks[-1] = chunks[-1].ljust(30, '0')  # padding '0' đủ 30

    return chunks, 30 - last_real_len