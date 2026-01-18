import pandas as pd #thư viện Pandas để xử lý dữ liệu dạng bảng
import matplotlib.pyplot as plt #thư viện vẽ biểu đồ
import seaborn as sns #thư viện vẽ biểu đồ nâng cao
from wordcloud import WordCloud #thư viện tạo WordCloud
import joblib # Dùng để lưu model lại, sau này dùng tiếp không cần train lại
import pyLDAvis
import pyLDAvis.lda_model # Thư viện tạo dashboard tương tác cho LDA
from sklearn.manifold import TSNE # Thuật toán giảm chiều dữ liệu để vẽ biểu đồ phân cụm
import warnings

warnings.filterwarnings("ignore")

# Cấu hình
INPUT_FILE = 'preprocessing.csv'
LDA_REPORT_FILE = 'lda_report.html'  # File kết quả tương tác cuối cùng
# Các file model đã lưu từ processing
MODEL_FILES = ['lda_model.pkl', 'vectorizer.pkl', 'data_vectorized.pkl']

# Cài đặt font chữ để hiển thị Tiếng Việt
plt.rcParams['figure.figsize'] = (12, 8) # Kích thước biểu đồ mặc định
try:
    plt.rcParams['font.family'] = 'Segoe UI'  # Font chuẩn trên Windows
except:
    pass  # Nếu không có thì dùng mặc định

# tải dữ liệu và model đã lưu
print("-> Đang tải dữ liệu và model...")
try:
    df = pd.read_csv(INPUT_FILE)
    # Chuyển cột ngày về dạng thời gian để vẽ biểu đồ xu hướng
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Load các dữ liệu đã nạp ở bước trước
    lda_model = joblib.load('lda_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    X = joblib.load('data_vectorized.pkl')
except Exception as e:
    print(f"LỖI: Không tìm thấy file dữ liệu hoặc model ({e}). Hãy chạy file xử lý trước.")
    exit()

# Biểu đồ Bar chart
# Mục đích: Xem chủ đề nào hot nhất, được báo chí viết nhiều nhất
print("-> Vẽ biểu đồ số lượng bài viết...")
plt.figure(figsize=(10, 6))# Vẽ biểu đồ cột ngang
sns.countplot(data=df, y='Topic_Name', order=df['Topic_Name'].value_counts().index, palette='viridis')
plt.title('Thống kê số lượng bài viết theo chủ đề')
plt.xlabel('Số lượng')
plt.tight_layout()
plt.show()

# Biểu đồ xu hướng theo thời gian
# Mục đích: Xem sự quan tâm của dư luận thay đổi thế nào theo thời gian
if 'date' in df.columns and df['date'].notna().any():
    print("3. Vẽ biểu đồ xu hướng theo thời gian...")
    # Gom nhóm dữ liệu theo Ngày và Chủ đề
    timeline = df.groupby([df['date'].dt.date, 'Topic_Name']).size().unstack(fill_value=0)

    timeline.plot(kind='line', marker='o', linewidth=2, figsize=(12, 6))# Vẽ biểu đồ đường
    plt.title('Diễn biến các chủ đề theo thời gian')
    plt.ylabel('Số bài viết')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Chủ đề')
    plt.tight_layout()
    plt.show()

# Biểu đồ từ khóa quan trọng trong mỗi chủ đề
# Mục đích: "Mổ xẻ" xem bên trong mỗi chủ đề, từ khóa nào đóng vai trò quyết định
print("-> Vẽ trọng số từ khóa quan trọng...")
feature_names = vectorizer.get_feature_names_out()
n_top_words = 10  # Lấy 10 từ quan trọng nhất
n_topics = len(lda_model.components_)

# Tạo lưới biểu đồ con
cols = 3
rows = (n_topics // cols) + (1 if n_topics % cols != 0 else 0)
fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
axes = axes.flatten()

for idx, topic in enumerate(lda_model.components_):
    # Lấy index của các từ có trọng số cao nhất
    top_indices = topic.argsort()[:-n_top_words - 1:-1]
    top_features = [feature_names[i] for i in top_indices]
    weights = topic[top_indices]

    # Vẽ cột ngang
    axes[idx].barh(top_features, weights, color='skyblue')
    axes[idx].set_title(f'Topic {idx}', fontweight='bold')
    axes[idx].invert_yaxis()  # Đảo ngược để từ quan trọng nhất lên đầu

# Xóa các ô trống nếu số biểu đồ không lấp đầy lưới
for i in range(idx + 1, len(axes)): fig.delaxes(axes[i])
plt.tight_layout()
plt.show()

# BẢN ĐỒ PHÂN CỤM (t-SNE)
# Mục đích: Gom nhóm bài viết lên mặt phẳng 2D.
# Nếu các cụm màu tách biệt rõ ràng -> Mô hình phân loại tốt.
print("->Đang chạy thuật toán t-SNE...")
# Perplexity: Tham số quan trọng, phải nhỏ hơn số lượng bài viết,(perplexity < num_samples / 3)
perp = min(30, len(df) - 1)
tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='random')# Giảm chiều dữ liệu về 2D
tsne_out = tsne.fit_transform(X.toarray())

plt.figure(figsize=(10, 8))
sns.scatterplot(x=tsne_out[:, 0], y=tsne_out[:, 1], hue=df['Topic_Name'], palette='deep', s=60, alpha=0.8)
plt.title('Bản đồ phân cụm bài viết (t-SNE)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Mây từ WordCloud cho mỗi chủ đề
# Mục đích: Trực quan hóa vui mắt, từ nào càng to thì càng xuất hiện nhiều
print("-> Vẽ WordCloud đại diện...")
for t_id in sorted(df['Topic_ID'].unique()):
    # Ghép toàn bộ văn bản thuộc chủ đề đó lại thành 1 chuỗi khổng lồ
    text = " ".join(df[df['Topic_ID'] == t_id]['clean_text'].astype(str))
    wc = WordCloud(width=800, height=300, background_color='white').generate(text)

    plt.figure(figsize=(6, 3))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Topic {t_id}")
    plt.show()

#  DASHBOARD TƯƠNG TÁC (pyLDAvis)
# Mục đích: Tạo báo cáo tương tác để khám phá chi tiết các chủ đề
print("-> Đang tạo báo cáo tương tác HTML...")
vis_data = pyLDAvis.lda_model.prepare(lda_model, X, vectorizer)
pyLDAvis.save_html(vis_data, LDA_REPORT_FILE)

print(f"-> Hoàn tất! Mở file {LDA_REPORT_FILE} để xem báo cáo ")