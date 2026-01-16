#Thuw viện và cấu hình chung cho việc crawl web
import requests                  # Thư viện để gửi yêu cầu HTTP (tải trang web)
from bs4 import BeautifulSoup    # Thư viện để phân tích cú pháp HTML
import pandas as pd              # Thư viện xử lý dữ liệu dạng bảng (DataFrame)
import time                      # Thư viện xử lý thời gian (ngủ, đo giờ)
import re                        # Thư viện xử lý biểu thức chính quy (Regex) - dùng để lọc ngày tháng
from datetime import datetime    # Thư viện xử lý định dạng ngày tháng

# Cấu hình Header để giả lập trình duyệt thật(tránh bị server chặn (lỗi 403 Forbidden)).
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


#Hàm lấy chi tiết bài viết và ngày đăng

def get_article_content(url):
    try:
        # Gửi request vào trang chi tiết
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        # Tìm thẻ chứa ngày đăng (Cấu trúc của VnExpress thường là span class="date")
        date_tag = soup.find("span", class_="date")
        date_str = date_tag.text.strip() if date_tag else ""

        # Xử lý chuỗi ngày tháng bằng Regex
        # Tìm chuỗi có dạng: ngày/tháng/năm (ví dụ: 2x/01/20xx)
        match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", date_str)
        publish_date = None

        if match:
            # Chuyển đổi chuỗi thành đối tượng datetime để dễ sắp xếp
            publish_date = datetime.strptime(match.group(1), "%d/%m/%Y")
        return publish_date
    except Exception as e:
        # Nếu có lỗi trả về None (ví dụ: lỗi kết nối, lỗi phân tích HTML)
        return None


# Hàm crawl chính

def scrape_real_news(base_category_url, target_count=300):
    data = []
    page = 1

    print(f"Đang thu thập dữ liệu từ: {base_category_url}")
    print(f"Mục tiêu: {target_count} bài viết.")

    while len(data) < target_count:
        # Xử lý phân trang (Pagination)
        # Trang 1 là link gốc, trang 2 trở đi thêm suffix "-p2", "-p3"...
        if page == 1:
            url = base_category_url
        else:
            url = f"{base_category_url}-p{page}"

        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            # Tìm tất cả thẻ chứa bài viết (class 'item-news' là đặc trưng của VnExpress)
            articles = soup.find_all("article", class_="item-news")

            if not articles:
                print("Không tìm thấy bài viết hoặc đã hết trang.")
                break

            for article in articles:
                # Kiểm tra số lượng
                if len(data) >= target_count:
                    break

                # Lấy tiêu đề và link
                title_tag = article.find("h3", class_="title-news")
                if not title_tag or not title_tag.find("a"):
                    continue

                title = title_tag.find("a").text.strip()
                link = title_tag.find("a")["href"]

                # Lọc bỏ Video/Podcast vì cấu trúc HTML khác biệt
                if "video" in link or "podcast" in link:
                    continue

                # Lấy mô tả ngắn (Sapo)
                desc_tag = article.find("p", class_="description")
                # Một số bài sapo nằm trong thẻ <a> bên trong <p> => check
                sapo = desc_tag.find("a").text.strip() if desc_tag and desc_tag.find("a") else ""
                # *lấy ngày đăng bài
                # Gọi hàm lấy ngay đăng để lấy ngày
                publish_date = get_article_content(link)

                # Chỉ lưu nếu có đủ thông tin quan trọng
                if title and sapo and publish_date:
                    data.append({
                        "date": publish_date,
                        "title": title,
                        "sapo": sapo,
                        "content_text": title + " " + sapo,  # Gộp text để tiện cho NLP sau này
                        "link": link  # Nên lưu link để tra cứu lại nếu cần
                    })

                    # In tiến độ ra màn hình
                    if len(data) % 10 == 0:
                        print(f"Đã lấy: {len(data)}/{target_count} bài.")

                # Rate limiting để tránh bị chặn IP nếu cào quá nhanh
                time.sleep(0.1)

            page += 1  # Chuyển sang trang tiếp theo

        except Exception as e:
            print(f"Lỗi tại trang {page}: {e}")
            break

    print(f"\n Hoàn thành! Tổng số bài lấy được: {len(data)}")
    return pd.DataFrame(data)

#Chạy hàm crawl và lưu dữ liệu
# Cấu hình tham số chạy
URL_CAN_LAY = "https://vnexpress.net/the-thao"  # Đổi link chuyên mục tại đây
SO_LUONG = 300

#  Gọi hàm thực thi
df = scrape_real_news(URL_CAN_LAY, target_count=SO_LUONG)

#  Sắp xếp dữ liệu theo thời gian (Mới nhất -> Cũ nhất hoặc ngược lại)
df = df.sort_values(by='date', ascending=False)

#  Hiển thị 5 dòng đầu tiên để kiểm tra
print("\n Dữ liệu mẫu ")
print(df.head())  # Trong Jupyter, dùng display() đẹp hơn print()

#  Lưu ra file CSV
file_name = 'dataset.csv'
df.to_csv(file_name, index=False, encoding='utf-8-sig') # utf-8-sig để mở bằng Excel không lỗi font tiếng Việt

print(f"\n lưu file: {file_name}")
# kiểm tra thông tin file
df.info()