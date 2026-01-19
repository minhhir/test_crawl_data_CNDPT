import requests # thư viện gửi yêu cầu HTTP
from bs4 import BeautifulSoup #thư viện BeautifulSoup để phân tích HTML
import pandas as pd #thư viện Pandas để xử lý dữ liệu dạng bảng
import time, random, re #thư viện hỗ trợ
from datetime import datetime #thư viện xử lý ngày tháng
from tqdm import tqdm #thư viện hiển thị thanh tiến trình

# --- CẤU HÌNH ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Hàm lấy nội dung chi tiết bài báo
def get_article_content(session, url):
    try:
        response = session.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        # 1. Xử lý ngày đăng (Tìm thẻ date -> Chuẩn hóa về YYYY-MM-DD)
        date_tag = soup.find("span", class_="date")
        date_str = date_tag.text.strip() if date_tag else ""
        try:
            match = re.search(r'\d{1,2}/\d{1,2}/\d{4}', date_str)
            clean_date = datetime.strptime(match.group(), '%d/%m/%Y').strftime(
                '%Y-%m-%d') if match else datetime.now().strftime('%Y-%m-%d')
        except:
            clean_date = datetime.now().strftime('%Y-%m-%d')

        # 2. Lấy Tiêu đề và Tóm tắt (Sapo)
        title = soup.find("h1", class_="title-detail").text.strip() if soup.find("h1", class_="title-detail") else ""
        sapo = soup.find("p", class_="description").text.strip() if soup.find("p", class_="description") else ""

        # 3. Lấy Nội dung chính (Gộp các thẻ p có class Normal)
        content_tags = soup.find_all("p", class_="Normal")#nếu là báo khác thì sửa class tương ứng
        body_text = " ".join([tag.text.strip() for tag in content_tags])

        # Nếu bài ảnh/video không có text class Normal, dùng tạm Sapo
        if not body_text: body_text = sapo

        return clean_date, title, sapo, body_text

    except Exception as e:
        return None, None, None, None

# Hàm chính để cào dữ liệu từ vnexpress
def scrape_data(base_url, target_count=300):#target_count: Số bài tự dộng cào nếu không nêu rõ số lượng
    data = []
    page = 1
    session = requests.Session()  # Tối ưu: Dùng session để giữ kết nối ổn định

    print(f"--- Bắt đầu cào dữ liệu: {base_url} ---")
    pbar = tqdm(total=target_count, desc="Tiến độ", unit="bài")

    while len(data) < target_count:
        # Tạo link phân trang (trang 1, trang 2...)
        url = f"{base_url}-p{page}" if page > 1 else base_url

        try:
            response = session.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            articles = soup.find_all("h3", class_="title-news")

            if not articles: break  # Hết bài thì dừng

            for article in articles:
                if len(data) >= target_count: break

                a_tag = article.find("a")
                if not a_tag: continue

                link = a_tag.get("href")
                if "video" in link or "podcast" in link: continue  # Bỏ qua video

                # Vào chi tiết lấy nội dung
                date, title, sapo, body = get_article_content(session, link)

                if title and body:
                    data.append({
                        "date": date,
                        "title": title,
                        "sapo": sapo,
                        "content_text": f"{title} {sapo} {body}",  # Dữ liệu sạch để tiền xử lý sau này
                        "link": link
                    })
                    pbar.update(1)

                # ngừng ngẫu nhiên để tránh bị chặn IP
                time.sleep(random.uniform(0.5, 1.2))

            page += 1  # Sang trang tiếp theo

        except Exception as e:
            print(f"Lỗi tại trang {page}: {e}")
            break

    pbar.close()

    # Lưu file CSV (encoding utf-8-sig để Excel đọc được tiếng Việt)
    pd.DataFrame(data).to_csv("dataset.csv", index=False, encoding="utf-8-sig")
    print(f" Đã lưu {len(data)} bài vào file 'dataset.csv'")


if __name__ == "__main__":
    # nếu file dataset.csv đã tồn tại thì xóa đi để cào lại
    import os
    if os.path.exists("dataset.csv"):
        os.remove("dataset.csv")
    #cào dữ liệu
    scrape_data("https://vnexpress.net/thoi-su", target_count=300) #target_count: Số bài muốn cào