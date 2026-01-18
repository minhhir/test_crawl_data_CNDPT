import pandas as pd# thư viện Pandas để xử lý dữ liệu dạng bảng
import re #thư viện hỗ trợ xử lý chuỗi
import os# thư viện hỗ trợ làm việc với file và đường dẫn
import joblib  # Dùng để lưu model lại, sau này dùng tiếp không cần train lại
from underthesea import word_tokenize  # Thư viện tách từ tiếng Việt tốt nhất hiện nay
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # thư viện chuyển văn bản thành dạng số
from sklearn.decomposition import LatentDirichletAllocation  # Thuật toán LDA để tìm chủ đề

# --- Cấu Hình ---
INPUT_FILE = 'dataset.csv'  # File dữ liệu đầu vào
OUTPUT_FILE = 'preprocessing.csv'  # File kết quả đầu ra
NUM_TOPICS = 8  # Số lượng chủ đề muốn máy tìm
STOPWORDS = set()  # Tập hợp các từ vô nghĩa (thì, là, mà...)
STOP_FILES = ['vietnamese-stopwords.txt']  # File chứa danh sách từ vô nghĩa

# Nạp danh sách từ vô nghĩa (nếu có file)
for f in STOP_FILES:
    if os.path.exists(f):
        try:
            content = open(f, encoding='utf-8').read().splitlines()
            STOPWORDS.update(content)
        except:
            pass


# Hàm làm sạch và tách từ
def preprocess_text(text):
    if not isinstance(text, str): return ""  # Kiểm tra nếu không phải chữ thì bỏ qua

    # 1. Chuyển về chữ thường để "Hà Nội" và "hà nội" là một
    text = text.lower()

    # 2. Xóa các ký tự đặc biệt (!@#$%), chỉ giữ lại chữ cái và số
    text = re.sub(r'[^\w\s]', '', text)

    # 3. Tách từ thông minh (Quan trọng với tiếng Việt)
    # Ví dụ: "đất nước" -> "đất_nước" (gộp thành 1 từ có nghĩa)
    tokens = word_tokenize(text, format="text").split()

    # 4. Lọc bỏ từ vô nghĩa (stopwords) và từ quá ngắn (1 ký tự)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    return " ".join(tokens)


# chạy chương trình
print("-> Đang đọc dữ liệu...")
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=['content_text'])  # Xóa dòng bị trống nội dung

print("-> Đang tiền xử lý (Làm sạch & Tách từ)...")
# Áp dụng hàm preprocess_text ở trên cho từng dòng
df['clean_text'] = df['content_text'].apply(preprocess_text)
# Xóa những dòng sau khi làm sạch thì không còn gì (ví dụ toàn icon)
df = df[df['clean_text'].str.strip() != ""]

# --- KỸ THUẬT HYBRID ---
# Tại sao dùng 2 cái Vectorizer?
# - TfidfVectorizer: Giỏi việc TÌM ra từ quan trọng, lọc bỏ từ rác xuất hiện quá nhiều.
# - CountVectorizer: Giỏi việc ĐẾM số lần xuất hiện (thích hợp cho LDA ).

print("-> Dùng TF-IDF để lọc từ quan trọng ...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=500,  # Chỉ lấy 1500 từ quan trọng nhất
    min_df=10,  # Từ phải xuất hiện ít nhất trong 10 bài mới tính (bỏ từ sai chính tả, quá hiếm)
    max_df=0.5  # Bỏ từ xuất hiện trong hơn 50% số bài (từ quá phổ thông như 'hôm nay', 'người dân')
)
tfidf_vectorizer.fit(df['clean_text']) # Học bộ từ vựng từ dữ liệu
top_vocabulary = tfidf_vectorizer.get_feature_names_out()  # Lấy danh sách từ vựng đã lọc

print("-> Chuyển dữ liệu sang dạng số đếm cho LDA...")
# Dùng bộ từ đã lọc ở trên để đếm
vectorizer = CountVectorizer(vocabulary=top_vocabulary)
X = vectorizer.fit_transform(df['clean_text'])

print(f"-> Đang chạy mô hình LDA => tìm {NUM_TOPICS} chủ đề...")
lda_model = LatentDirichletAllocation(
    n_components=NUM_TOPICS,  # Số chủ đề
    max_iter=50,  # Số lần học lặp lại (càng cao càng kỹ nhưng lâu)
    learning_method='batch',  # Học theo lô (với dữ liệu nhỏ=> cho hiệu quả hơn)
    random_state=42,  # Giữ cố định kết quả mỗi lần chạy
    learning_offset=50.0,  # Tham số ổn định (giá trị lớn hơn 1.0)
    evaluate_every=-1,  # Không đánh giá trong quá trình học (tiết kim thời gian)
    verbose=1,  # Hiện tiến trình học
    doc_topic_prior=0.1,# Mức độ tập trung chủ đề trong mỗi tài liệu
    topic_word_prior=0.01,# Mức độ tập trung từ trong mỗi chủ đề
    n_jobs=-1  # Dùng tất cả nhân CPU để chạy cho nhanh
)
lda_model.fit(X)

# Lưu model và dữ liệu đã xử lý
print("->Đang lưu model...")
joblib.dump(vectorizer, 'vectorizer.pkl')  # Lưu bộ đếm
joblib.dump(lda_model, 'lda_model.pkl')  # Lưu trí tuệ của AI
joblib.dump(X, 'data_vectorized.pkl')  # Lưu dữ liệu đã số hóa

print("-> Đang đặt tên cho chủ đề...")
feature_names = vectorizer.get_feature_names_out()
topic_names = {}
topic_keywords = {}

# Duyệt qua từng chủ đề tìm được để đặt tên
for topic_idx, topic in enumerate(lda_model.components_):
    # Lấy 3 từ trọng số cao nhất để ghép thành tên chủ đề
    top_3_ind = topic.argsort()[:-4:-1]
    name_parts = [feature_names[i].replace("_", " ").title() for i in top_3_ind]
    topic_names[topic_idx] = " - ".join(name_parts)

    # Lấy 10 từ khóa đặc trưng nhất để mô tả
    top_10 = topic.argsort()[:-11:-1]
    topic_keywords[topic_idx] = ", ".join([feature_names[i] for i in top_10])

# Gán ngược lại vào file Excel
topic_values = lda_model.transform(X)
df['Topic_ID'] = topic_values.argmax(axis=1)  # Bài này nghiêng về chủ đề nào nhất?
df['Topic_Name'] = df['Topic_ID'].map(topic_names)# Gán tên chủ đề
df['Topic_Keywords'] = df['Topic_ID'].map(topic_keywords)# Gán từ khóa chủ đề

df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"-> Hoàn tất! Kết quả đã lưu vào '{OUTPUT_FILE}'")