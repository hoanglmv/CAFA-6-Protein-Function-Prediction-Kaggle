import torch
import torch.nn as nn
import time

# --- BƯỚC 1: CẤU HÌNH THIẾT BỊ ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("-" * 30)
# SỬA LỖI TẠI ĐÂY: Thêm str() để biến device thành chuỗi
print(f"Đang chạy trên thiết bị: {str(device).upper()}")

if device.type == 'cuda':
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
    # Hàm lấy VRAM cũng cần ép kiểu số để tính toán
    print(f"VRAM hiện tại: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
print("-" * 30)

# --- BƯỚC 2: TẠO DỮ LIỆU GIẢ (RANDOM) ---
input_size = 1000
output_size = 10
data_count = 10000

print("Đang tạo dữ liệu và đẩy vào GPU...")
X = torch.randn(data_count, input_size).to(device)
y = torch.randn(data_count, output_size).to(device)

# --- BƯỚC 3: XÂY DỰNG MODEL ---
model = nn.Sequential(
    nn.Linear(input_size, 4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, output_size)
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- BƯỚC 4: QUÁ TRÌNH TRAIN ---
print("\nBắt đầu Training...")
start_time = time.time()
epochs = 100

for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

end_time = time.time()
print("-" * 30)
print(f"Hoàn thành trong: {end_time - start_time:.2f} giây")
print("Đã train thành công trên GPU!")