import torch

# 1. Load checkpoint
ckpt = torch.load('./latest_G_1.pth', map_location='cpu')
state_dict = ckpt.get('model_state_dict', ckpt)

# 2. Mở file để ghi (UTF-8 để hỗ trợ Unicode)
with open('num_params_all.txt', 'w', encoding='utf-8') as f:
    total_params = 0
    
    # Header
    f.write(f"{'Layer name':50s} {'Shape':25s} {'#Params':10s}\n")
    f.write("="*90 + "\n")
    
    # 3. Duyệt qua từng tensor, tính và ghi vào file
    for name, tensor in state_dict.items():
        num = tensor.numel()  # số phần tử = tích các chiều
        total_params += num
        f.write(f"{name:50s} {str(tuple(tensor.shape)):25s} {num:10d}\n")
    
    # 4. Ghi tổng số parameter và phần giải thích
    f.write("\n")
    f.write(f"Tổng số parameter của bitencoder: {total_params}\n\n")
    f.write("Giải thích:\n")
    f.write("- Với mỗi tensor, #Params = tích các dimensions của nó (ví dụ (16,3,3,3) → 16×3×3×3 = 432).\n")
    f.write("- Tổng số parameter là tổng #Params của tất cả các tensor trong mô hình.\n")
    f.write("- Đây chính là số lượng biến cần học (weights và biases) của phần bitencoder.\n")
