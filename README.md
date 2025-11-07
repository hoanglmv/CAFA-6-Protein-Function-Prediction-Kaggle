Đây là cấu trúc folder data.csvới mô tả chi tiết từng file:
Bạn hãy giúp tôi code các file sau vào trong thư mục /src
viewer.ipynb : giúp tôi xem và phân tích data
process_data.ipynb : giúp tôi xử lý dữ liệu để đưa vào mô hình học máy
model_training.ipynb : giúp tôi xây dựng và huấn luyện mô hình học máy
evaluation.ipynb : giúp tôi đánh giá mô hình học máy sau khi huấn luyện x

1. train_sequences.fasta

Chứa trình tự amino acid của các protein trong tập huấn luyện.

Định dạng FASTA:

>Protein_ID
MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAY


Dòng bắt đầu bằng > là ID của protein.

Các dòng tiếp theo là chuỗi amino acid.

Đây là feature chính (input) bạn sẽ dùng để học mô hình dự đoán chức năng.

2. train_terms.tsv

Liên kết giữa Protein ID và các GO terms (chức năng).

Cấu trúc thường là:

Protein_ID   GO_ID
A0A024RBG1   GO:0005515
A0A024RBG1   GO:0000166
A0A023SWA7   GO:0003677


Mỗi hàng là một cặp: một protein – một chức năng GO term.

Một protein có thể có nhiều dòng tương ứng với nhiều GO term → multi-label classification.

Đây chính là target (y) của mô hình.

3. train_taxonomy.tsv

Cung cấp thông tin loài (taxon ID) cho mỗi protein trong tập train.

Dạng thường thấy:

Protein_ID   Taxon_ID
A0A024RBG1   9606
A0A023SWA7   10090


9606 = người (Homo sapiens), 10090 = chuột (Mus musculus), …

Có thể dùng làm feature phụ vì protein có cùng loài thường chia sẻ chức năng tương tự.

4. go-basic.obo

File định nghĩa Gene Ontology (GO) — hệ thống phân loại chức năng protein.

Gồm ba nhánh chính:

MF (Molecular Function) — ví dụ: binding, catalytic activity.

BP (Biological Process) — ví dụ: cell division, metabolism.

CC (Cellular Component) — ví dụ: nucleus, membrane.

Mỗi term có thông tin:

[Term]
id: GO:0005515
name: protein binding
namespace: molecular_function
is_a: GO:0005488 ! binding


Dùng để:

Hiểu ngữ nghĩa và quan hệ cha-con giữa các GO terms.

Hỗ trợ lan truyền nhãn (gán parent term khi con xuất hiện).

Tính độ tương tự chức năng (semantic similarity).

🧫 Thư mục Test/

Dữ liệu cần dự đoán (không có nhãn thật).

5. testsuperset.fasta

Cấu trúc giống train_sequences.fasta.

Chứa trình tự amino acid của các protein cần dự đoán GO terms.

Bạn cần dự đoán chức năng (GO terms) cho các protein này → tạo submission.

6. testsuperset-taxon-list.tsv

Giống train_taxonomy.tsv nhưng dành cho tập test.

Cho bạn biết mỗi protein trong test thuộc loài nào.

Hữu ích nếu bạn muốn tận dụng thông tin loài khi dự đoán.

📄 Các file khác
7. sample_submission.tsv

Mẫu file kết quả mà bạn cần nộp lên Kaggle.

Dạng thường thấy:

Protein_ID   GO_ID
A0A024RBG1   GO:0005515
A0A024RBG1   GO:0000166
...


Bạn cần điền vào đây dự đoán GO terms cho từng protein trong test.

8. IA.tsv

Thường là file thống kê hoặc mapping nội bộ (chẳng hạn Information Accretion hoặc ID alignment).

Một số notebook dùng nó để tính toán metric hoặc xử lý nhãn đặc biệt.

Nếu không có hướng dẫn cụ thể, bạn có thể tạm bỏ qua cho đến khi dùng trong phần đánh giá.