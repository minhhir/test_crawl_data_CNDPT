import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- GIẢI THÍCH CÔNG DỤNG THƯ VIỆN ---
# 1. Matplotlib (plt): Thư viện vẽ biểu đồ nền tảng, cơ bản nhất của Python.
#    Nó giống như "giấy và bút chì", cho phép bạn vẽ mọi thứ từ đường thẳng đến biểu đồ phức tạp.
#
# 2. Seaborn (sns): Được xây dựng dựa trên Matplotlib nhưng "đẹp hơn" và "dễ dùng hơn".
#    Nó giúp vẽ các biểu đồ thống kê màu sắc đẹp mắt mà không cần code quá nhiều dòng.
#
# 3. WordCloud: Công cụ chuyên dụng để tạo "đám mây từ khóa".
#    Từ nào xuất hiện càng nhiều thì kích thước hiển thị càng to.

# --- CẤU HÌNH ĐẦU VÀO ---
INPUT_FILE = 'preprocessing.csv'  # File csv kết quả từ bước LDA trước đó
COLUMN_DATE = 'date'              # Tên cột ngày tháng (cần thay đổi nếu file bạn tên khác)


# 1. ĐỌC DỮ LIỆU
print(f" Đang đọc dữ liệu từ '{INPUT_FILE}'...")
try:
    df = pd.read_csv(INPUT_FILE)

    # Kiểm tra các cột bắt buộc
    required_cols = ['Topic_ID', 'Topic_Keywords', 'clean_text', COLUMN_DATE]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"File CSV thiếu một trong các cột bắt buộc: {required_cols}")

    # Xử lý cột ngày tháng (Chuyển từ chuỗi sang datetime để vẽ biểu đồ đường)
    df['date_obj'] = pd.to_datetime(df[COLUMN_DATE], errors='coerce')

    # Tạo lại biến keywords_map từ dữ liệu trong file CSV để dùng cho việc gắn nhãn
    # (Biến chuỗi 'Topic_ID' và 'Topic_Keywords' thành từ điển)
    keywords_map = dict(zip(df.Topic_ID, df.Topic_Keywords))

    NUM_TOPICS = df['Topic_ID'].nunique()  # Đếm số lượng chủ đề có trong file

except Exception as e:
    print(f" Lỗi đọc file: {e}")
    exit()

# 2. CẤU HÌNH FONT (Tránh lỗi tiếng Việt)
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']

# --- BIỂU ĐỒ 1: PIE CHART (Tỷ trọng) ---
print("Đang vẽ biểu đồ tròn...")
plt.figure(figsize=(8, 8))
topic_counts = df['Topic_ID'].value_counts()
plt.pie(topic_counts, labels=[f"Chủ đề {i}" for i in topic_counts.index],
        autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title(f"Tỷ lệ phân bố {NUM_TOPICS} chủ đề", fontsize=14)
plt.show()

# --- BIỂU ĐỒ 2: BAR CHART (Số lượng + Từ khóa) ---
print("Đang vẽ biểu đồ cột...")
plt.figure(figsize=(12, 6))
ax = sns.countplot(x='Topic_ID', data=df, hue='Topic_ID', palette='viridis', legend=False)
plt.title("Số lượng bài viết trong từng nhóm chủ đề", fontsize=14)
plt.xlabel("Mã chủ đề")
plt.ylabel("Số bài viết")

# In từ khóa lên đầu cột
for p, label in zip(ax.patches, topic_counts.index):
    if label in keywords_map:
        # Lấy 2 từ khóa đầu tiên để hiển thị cho gọn
        full_kw = str(keywords_map[label])
        kw_short = ", ".join(full_kw.split(',')[:2])

        ax.annotate(f"{kw_short}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontsize=9)
plt.tight_layout()
plt.show()

# --- BIỂU ĐỒ 3: LINE CHART (Xu hướng thời gian) ---
print("Đang vẽ biểu đồ đường...")
# Loại bỏ dòng không có ngày tháng hợp lệ
df_time = df.dropna(subset=['date_obj'])

if not df_time.empty:
    timeline_df = df_time.groupby([df_time['date_obj'].dt.date, 'Topic_ID']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    for topic_id in timeline_df.columns:
        plt.plot(timeline_df.index, timeline_df[topic_id], marker='o', label=f"Chủ đề {topic_id}")

    plt.title("Xu hướng thay đổi theo thời gian", fontsize=14)
    plt.xlabel("Thời gian")
    plt.ylabel("Số lượng tin bài")
    plt.legend(title="Chủ đề", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Không thể vẽ biểu đồ đường vì cột ngày tháng bị rỗng hoặc sai định dạng.")

# --- BIỂU ĐỒ 4: WORD CLOUD ---
print("Đang tạo Word Cloud...")
# Tìm chủ đề lớn nhất
top_topic_id = topic_counts.idxmax()
# Gom văn bản của chủ đề đó lại
text_corpus = " ".join(df[df['Topic_ID'] == top_topic_id]['clean_text'].astype(str))

# Lưu ý: Cần file font .ttf nếu muốn hiển thị tiếng Việt chuẩn trên WordCloud
# Ví dụ: font_path='C:/Windows/Fonts/arial.ttf'
wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text_corpus)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title(f"WordCloud: Chủ đề phổ biến nhất (ID: {top_topic_id})", fontsize=14)
plt.show()

print("Hoàn tất!")