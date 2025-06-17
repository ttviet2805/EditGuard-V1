#!/usr/bin/env python3
import os
from pathlib import Path

# Thư mục cần liệt kê
DIR = Path('COCO-2017-1000-images')

# File đầu ra
OUTPUT = DIR / 'COCO-2017-1000-images.txt'

def main():
    # Lấy danh sách file, chỉ file (không tính thư mục con), sắp xếp tên
    file_names = sorted([p.name for p in DIR.iterdir() if p.is_file()])

    # Ghi ra file
    with OUTPUT.open('w', encoding='utf-8') as f:
        for name in file_names:
            f.write(name + '\n')

    print(f"✅ Đã ghi {len(file_names)} tên file vào {OUTPUT}")

if __name__ == "__main__":
    main()
