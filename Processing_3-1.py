#Thư viện và cấu hình chung cho làm sạch dữ liệu văn bản tiếng Việt và Topic Modeling

import pandas as pd  # Thư viện xử lý dữ liệu dạng bảng (DataFrame), giúp đọc/ghi file CSV dễ dàng

import re  # (Regular Expression) Dùng để tìm kiếm, thay thế ký tự lạ, làm sạch văn bản

from underthesea import word_tokenize #  Dùng để tách từ tiếng Việt (Word Segmentation)-Quan trọng: Giúp máy hiểu "nhà máy" là 1 từ, thay vì 2 từ rời rạc "nhà" và "máy".

# TfidfVectorizer: Chuyển đổi văn bản thô thành ma trận số liệu thống kê (TF-IDF)-Giúp xác định từ nào quan trọng trong văn bản (xuất hiện nhiều nhưng không đại trà).
from sklearn.feature_extraction.text import TfidfVectorizer

# LatentDirichletAllocation (LDA): Thuật toán Topic Modeling-Dùng để gom nhóm các văn bản có cấu trúc từ giống nhau vào cùng một chủ đề.
from sklearn.decomposition import LatentDirichletAllocation

INPUT_FILE = 'dataset.csv'        # Tên file CSV đầu vào
OUTPUT_FILE = 'preprocessing.csv' # Tên file kết quả đầu ra
COLUMN_NAME = 'content_text'      # Tên cột chứa dữ liệu văn bản

# Danh sách Stopwords tiếng Việt-Đây là các từ không mang nhiều ý nghĩa phân loại chủ đề
STOPWORDS = {
    'là', 'của', 'và', 'các', 'những', 'trong', 'với', 'cho', 'người', 'được',
    'khi', 'đã', 'sẽ', 'đang', 'về', 'ở', 'làm', 'ra', 'này', 'cũng', 'đến',
    'từ', 'có', 'không', 'như', 'để', 'một', 'nhiều', 'theo', 'nhưng', 'bị',
    'vì', 'tại', 'vào', 'do', 'lên', 'xuống', 'trên', 'dưới', 'ngày', 'tháng', 'năm',
    'rằng', 'thì', 'mà'  # Bổ sung thêm vài từ phổ biến
}

# Hàm làm sạch text
def preprocess_text(text):
    if not isinstance(text, str):  # Kiểm tra nếu giá trị là NaN hoặc không phải chuỗi
        return ""

    # Chuyển thành chữ thường
    text = text.lower()

    # Loại bỏ ký tự đặc biệt (giữ lại chữ cái và số)-Regex: [^\w\s] nghĩa là những gì không phải chữ (word) và khoảng trắng (space)
    text = re.sub(r'[^\w\s]', '', text)

    # Tách từ tiếng Việt bằng underthesea (format="text" để nối từ ghép bằng dấu gạch dưới) Ví dụ: "học sinh" -> "học_sinh"
    tokens = word_tokenize(text, format="text").split()

    # Loại bỏ stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

#Đọc và kiểm tra dữ liệu
try:
    df = pd.read_csv(INPUT_FILE)

    # Kiểm tra cột dữ liệu
    if COLUMN_NAME not in df.columns:
        raise ValueError(f"Lỗi: Không tìm thấy cột '{COLUMN_NAME}' trong file CSV.")

    # Loại bỏ các dòng trống (NaN) ngay từ đầu
    df = df.dropna(subset=[COLUMN_NAME])
    print(f"Số lượng dòng dữ liệu: {len(df)}")

    # Hiển thị 5 dòng đầu tiên để kiểm tra
    print(df.head())

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{INPUT_FILE}'.")

# Tiền xử lý dữ liệu văn bản

# Áp dụng hàm preprocess_text cho cột nội dung
df['clean_text'] = df[COLUMN_NAME].apply(preprocess_text)

# Loại bỏ những dòng sau khi clean mà bị rỗng (ví dụ dòng chỉ chứa icon hoặc stopword)
df = df[df['clean_text'].str.strip() != ""]
print(df[['clean_text']].head()) # Xem kết quả sau khi làm sạch

#Vectorize văn bản bằng TF-IDF

# Cấu hình TF-IDF:
# - max_features=1500: Chỉ lấy 1500 từ quan trọng nhất
# - min_df=5: Từ phải xuất hiện ít nhất trong 5 văn bản
# - max_df=0.9: Loại bỏ từ xuất hiện quá nhiều (trên 90% văn bản - thường là từ phổ thông)
vectorizer = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.9)
X = vectorizer.fit_transform(df['clean_text'])
print(f"Kích thước ma trận TF-IDF: {X.shape}")

# Xây dựng mô hình LDA để phân nhóm chủ đề
NUM_TOPICS = 5 # Số lượng chủ đề muốn tìm

# Khởi tạo mô hình LDA
lda_model = LatentDirichletAllocation(
    n_components=NUM_TOPICS, # Số chủ đề
    random_state=42, # Đặt seed để kết quả có thể tái lập
    learning_method='online', # 'online' thường nhanh hơn cho dữ liệu lớn
    n_jobs=-1 # Sử dụng tất cả CPU core
)
lda_model.fit(X)# Huấn luyện mô hình LDA trên ma trận TF-IDF

# Gán chủ đề cho từng văn bản
topic_results = lda_model.transform(X)# Lấy xác suất chủ đề cho từng văn bản
df['Topic_ID'] = topic_results.argmax(axis=1) # Lấy index của chủ đề có xác suất cao nhất

# Hàm lấy từ khóa đặc trưng cho từng chủ đề
def get_topic_keywords(model, feature_names, n_top_words=5):
    topic_keywords = {}
    for topic_idx, topic in enumerate(model.components_):
        # Sắp xếp và lấy index của n từ có trọng số cao nhất
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        keywords = [feature_names[i] for i in top_features_ind]
        topic_keywords[topic_idx] = ", ".join(keywords)
    return topic_keywords

# Lấy danh sách từ khóa từ vectorizer
feature_names = vectorizer.get_feature_names_out()
keywords_map = get_topic_keywords(lda_model, feature_names)

# Map từ khóa vào DataFrame
df['Topic_Keywords'] = df['Topic_ID'].map(keywords_map)

print("Kết quả phân loại chủ đề:")
print(df[[COLUMN_NAME, 'Topic_ID', 'Topic_Keywords']].head())

# Lưu kết quả ra file CSV
# encoding='utf-8-sig' rất quan trọng để Excel mở tiếng Việt không bị lỗi font
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

