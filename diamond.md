# Hướng dẫn tạo Diamond Matches (Homology Search)

Tài liệu này hướng dẫn cách tạo file `diamond_matches.tsv` sử dụng công cụ **Diamond** để tìm kiếm sự tương đồng trình tự (sequence alignment) giữa tập Test và tập Train. Kết quả này được sử dụng để ensemble với mô hình Deep Learning.

## 1. Cài đặt Diamond

Sử dụng Conda hoặc APT:

```bash
conda install -c bioconda diamond
# Hoặc trên Ubuntu
sudo apt-get install diamond-aligner
```

## 2. Chuẩn bị dữ liệu

Đảm bảo các file sau tồn tại đúng vị trí:
- **Train Sequences:** `data/Train/train_sequences.fasta` (Dữ liệu dùng để xây dựng Database)
- **Test Sequences:** `data/Test/testsuperset.fasta` (Dữ liệu cần truy vấn)

## 3. Thực thi

### Bước 1: Tạo Database (MakeDB)
Lệnh này chuyển đổi file FASTA của tập Train sang định dạng nhị phân của Diamond.

```bash
diamond makedb \
    --in data/Train/train_sequences.fasta \
    --db data/Train/train_db
```

### Bước 2: Chạy Alignment (Blastp)
So khớp tập Test với Database vừa tạo.

**Giải thích tham số:**
- `--outfmt 6`: Định dạng output chuẩn BLAST (Tab-separated). 3 cột đầu tiên mặc định là: `QueryID` (Test), `SubjectID` (Train), `PercentIdentity`.
- `--sensitivity very-sensitive`: Tăng độ nhạy để tìm được các protein xa hơn (tốn thời gian hơn chút nhưng tốt cho CAFA).
- `--max-target-seqs 1`: Chỉ lấy protein giống nhất trong tập Train cho mỗi protein Test.

```bash
diamond blastp \
    --db data/Train/train_db.dmnd \
    --query data/Test/testsuperset.fasta \
    --out data/processed2/diamond_matches.tsv \
    --outfmt 6 \
    --max-target-seqs 1
```

## 4. Kiểm tra kết quả

File output `data/processed2/diamond_matches.tsv` sẽ có dạng:

| TestID (qseqid) | TrainID (sseqid) | Pident | ... |
|-----------------|------------------|--------|-----|
| T12345          | sp\|P12345\|NAME | 98.5   | ... |

Code Python (`predict.py`) sẽ tự động xử lý việc cắt chuỗi `sp|...` trong cột TrainID.