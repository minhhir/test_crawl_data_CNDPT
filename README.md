# 📰 Khai Thác & Phân Tích Chủ Đề Tin Tức (News Topic Modeling)

Dự án này xây dựng một quy trình tự động để thu thập dữ liệu từ các trang tin tức trực tuyến, xử lý ngôn ngữ tiếng Việt và áp dụng thuật toán **LDA (Latent Dirichlet Allocation)** để phát hiện các chủ đề nổi bật.

## 🚀 Tính năng chính

* **Crawling:** Tự động thu thập dữ liệu (Title, Sapo, Content, Date) từ các trang báo điện tử.
* **Preprocessing:** Làm sạch văn bản, tách từ tiếng Việt (Word Tokenization) và loại bỏ từ dừng (Stopwords).
* **Modeling:** Phân lớp chủ đề tự động sử dụng thuật toán LDA.
* **Visualization:** Trực quan hóa kết quả bằng biểu đồ phân cụm (t-SNE), đám mây từ (WordCloud) và báo cáo tương tác (PyLDAvis).

## 📂 Cấu trúc dự án

| File | Mô tả |
| :--- | :--- |
| `test_Crawl_3-1.py` | Script thu thập dữ liệu. Tự động crawl và lưu kết quả vào `dataset.csv`. |
| `Processing_3-1.py` | Script tiền xử lý dữ liệu (dùng `underthesea`), huấn luyện mô hình LDA và lưu model (`.pkl`). |
| `Pt3_1.py` | Script trực quan hóa. Tạo biểu đồ t-SNE, WordCloud và xuất file báo cáo HTML. |
| `vietnamese-stopwords.txt` | Danh sách các từ dừng tiếng Việt cần loại bỏ. |
| `lda_report.html` | Báo cáo tương tác hiển thị phân phối các chủ đề (kết quả của PyLDAvis). |
| `dataset.csv` | Dữ liệu thô sau khi crawl. |
| `preprocessing.csv` | Dữ liệu sạch sau khi tiền xử lý. |

## 🛠️ Cài đặt & Yêu cầu

Dự án yêu cầu **Python 3.8+** và các thư viện sau:

```bash
pip install pandas scikit-learn underthesea matplotlib seaborn wordcloud pyldavis requests beautifulsoup4 tqdm
```
📖 Hướng dẫn sử dụng
Chạy lần lượt các bước sau để thực hiện quy trình khai thác dữ liệu:

Bước 1: Thu thập dữ liệu
Chạy file crawler để lấy bài viết mới nhất:

```Bash
python test_Crawl_3-1.py
```
python test_Crawl_3-1.py
Output: File dataset.csv

Bước 2: Xử lý & Huấn luyện mô hình
Làm sạch dữ liệu và training model LDA:

```Bash
python Processing_3-1.py
```
python Processing_3-1.py
Output: File preprocessing.csv, lda_model.pkl, vectorizer.pkl

Bước 3: Trực quan hóa kết quả
Vẽ biểu đồ và tạo báo cáo:

```Bash
python Pt3_1.py
```

python Pt3_1.py
Output: Hiển thị biểu đồ t-SNE, WordCloud và tạo file lda_report.html.

📊 Kết quả Demo
1. Phân cụm chủ đề (t-SNE)
(Bạn có thể chèn hình ảnh image_8ae52d.png vào đây để minh họa)

2. Dashboard tương tác (PyLDAvis)
Mở file lda_report.html trên trình duyệt để xem chi tiết các từ khóa trọng tâm và sự phân bổ của từng chủ đề.

Author: Hoàng Năng Minh
