#!/usr/bin/env python3
import os
from pathlib import Path

# Thay thành thư mục của bạn
DIR = Path('COCO-2017-1000-images')

def main():
    # 1) Lấy danh sách file (không bao gồm thư mục con), sắp xếp theo tên
    files = sorted([p for p in DIR.iterdir() if p.is_file()])

    N = len(files)
    if N == 0:
        print("Thư mục trống, không có file nào để đổi tên.")
        return

    # 2) Xác định width (tối thiểu 4 chữ số, tự mở rộng nếu N>9999)
    width = max(4, len(str(N)))

    # 3) Pass 1: đổi tên từng file → tên tạm ('0001.tmp', …) để tránh conflict
    temp_paths = []
    for idx, old_path in enumerate(files, start=1):
        tmp_name = f"{idx:0{width}d}.tmp"
        tmp_path = DIR / tmp_name
        old_path.rename(tmp_path)
        temp_paths.append(tmp_path)

    # 4) Pass 2: đổi tên từ .tmp → .png
    for tmp_path in temp_paths:
        stem = tmp_path.stem               # e.g. "0001"
        new_name = f"{stem}.png"
        new_path = DIR / new_name
        tmp_path.rename(new_path)

    print(f"✅ Đã đổi tên {N} file trong {DIR!r} thành 0001.png → {N:0{width}d}.png")

if __name__ == "__main__":
    main()
