from collections import defaultdict
import re

# Biến lưu tổng params theo module
params_by_module = defaultdict(int)

with open('num_params_all.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        # Bỏ qua header và các dòng phân cách
        if not line or line.startswith('Layer name') or line.startswith('==='):
            continue
        # Tách thành: tên_layer, shape, num_params
        parts = re.split(r'\s{2,}', line)
        if len(parts) < 3:
            continue
        name, shape, num = parts[0], parts[1], parts[2]
        try:
            num = int(num)
        except ValueError:
            continue
        # Lấy module chính (từ đầu đến dấu '.')
        module = name.split('.')[0]
        params_by_module[module] += num

# In kết quả
with open('module_params.txt', 'w', encoding='utf-8') as out:
    out.write("Module      \t#Params\n")
    out.write("======================\n")
    for module, total in params_by_module.items():
        out.write(f"{module:12s}\t{total}\n")

print("Đã ghi tổng số param theo module vào module_params.txt")
