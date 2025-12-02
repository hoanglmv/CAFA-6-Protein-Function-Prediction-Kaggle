import torch
import numpy as np
from transformers import AutoTokenizer, EsmModel
from ..utils.load_config import load_config

config = load_config()

model_name = config["esm"].get("model_name", "facebook/esm2_t36_3B_UR50D")
max_length = config["esm"].get("max_length", 1024)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EsmModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()


def get_protein_embedding(sequence):
    """
    Hàm encode một chuỗi protein đơn lẻ thành vector embedding sử dụng ESM-2.
    Áp dụng Mean Pooling.

    Args:
        sequence (str): Chuỗi acid amin (ví dụ: "MKTVRQ...").

    Returns:
        numpy.ndarray: Vector embedding 1 chiều.
    """

    safe_sequence = sequence.replace(" ", "")

    inputs = tokenizer(
        [safe_sequence],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    # 3. Inference (Không tính gradient)
    with torch.no_grad():
        outputs = model(**inputs)

        # Lấy hidden state cuối cùng: Shape (1, Sequence_Length, Hidden_Dim)
        last_hidden_states = outputs.last_hidden_state

    # 4. Mean Pooling (Logic chuẩn để loại bỏ padding và token đặc biệt)
    # Lấy mask (đánh dấu vị trí nào là dữ liệu thật, vị trí nào là pad)
    attention_mask = inputs["attention_mask"]

    # Mở rộng mask để khớp dimension với hidden states
    # Shape mask đang là (1, Seq_Len) -> đổi thành (1, Seq_Len, Hidden_Dim)
    mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    )

    # Nhân hidden state với mask (để biến các vị trí pad thành 0)
    sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)

    # Tính tổng số lượng token thật (tránh chia cho 0 bằng cách clamp min)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)

    # Chia trung bình
    mean_embedding = sum_embeddings / sum_mask

    # 5. Chuyển kết quả từ Tensor (GPU) về Numpy (CPU)
    # .squeeze() để bỏ chiều batch (1, dim) -> (dim,)
    return mean_embedding.cpu().numpy().squeeze()


# # ==========================================
# # VÍ DỤ SỬ DỤNG VỚI MODEL 3B VÀ LENGTH 4096
# # ==========================================
# if __name__ == "__main__":
#     # Cấu hình
#     # Lưu ý: Model 3B rất nặng, nếu máy yếu hãy đổi về "facebook/esm2_t33_650M_UR50D"
#     MODEL_NAME = "facebook/esm2_t36_3B_UR50D"
#     # MODEL_NAME = "facebook/esm2_t6_8M_UR50D" # Dùng cái này để test code cho nhanh

#     MAX_LEN = 4096  # Set theo yêu cầu của bạn

#     print("Đang load model...")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load model & tokenizer một lần duy nhất ở ngoài vòng lặp
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     model = EsmModel.from_pretrained(MODEL_NAME).to(device)
#     model.eval()

#     print(f"Load xong model trên thiết bị: {device}")

#     # Chuỗi ví dụ dài
#     my_protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" * 5

#     # Gọi hàm
#     vector = get_protein_embedding(
#         my_protein, model, tokenizer, device, max_length=MAX_LEN
#     )

#     print("-" * 30)
#     print(f"Input length: {len(my_protein)}")
#     print(f"Output embedding shape: {vector.shape}")
#     print(f"5 giá trị đầu: {vector[:5]}")

#     # Kiểm tra dimension:
#     # Model 8M   -> 320
#     # Model 650M -> 1280
#     # Model 3B   -> 2560
