import base64
import random


def img_to_base64(filepath):
    with open(filepath, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
def rand(num_bits=64):
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