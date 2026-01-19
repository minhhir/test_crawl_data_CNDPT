import requests  # thư viện gửi yêu cầu HTTP
from bs4 import BeautifulSoup  # thư viện BeautifulSoup để phân tích HTML
import pandas as pd  # thư viện Pandas để xử lý dữ liệu dạng bảng
import time, random, re, os  # thư viện hỗ trợ
from datetime import datetime  # thư viện xử lý ngày tháng
from tqdm import tqdm  # thư viện hiển thị thanh tiến trình

# --- CẤU HÌNH ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


# Hàm lấy nội dung chi tiết bài báo
def get_article_content(session, url):
    try:
        response = session.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200: return None, None, None, None

        soup = BeautifulSoup(response.content, "html.parser")

        # 1. Xử lý ngày đăng
        date_tag = soup.find("span", class_="date")
        date_str = date_tag.text.strip() if date_tag else ""
        try:
            match = re.search(r'\d{1,2}/\d{1,2}/\d{4}', date_str)
            clean_date = datetime.strptime(match.group(), '%d/%m/%Y').strftime(
                '%Y-%m-%d') if match else datetime.now().strftime('%Y-%m-%d')
        except:
            clean_date = datetime.now().strftime('%Y-%m-%d')

        # 2. Lấy Tiêu đề và Tóm tắt
        title = soup.find("h1", class_="title-detail").text.strip() if soup.find("h1", class_="title-detail") else ""
        sapo = soup.find("p", class_="description").text.strip() if soup.find("p", class_="description") else ""

        # 3. Lấy Nội dung chính
        content_tags = soup.find_all("p", class_="Normal")
        body_text = " ".join([tag.text.strip() for tag in content_tags])

        # Fallback nếu không có nội dung class Normal
        if not body_text: body_text = sapo

        return clean_date, title, sapo, body_text

    except Exception as e:
        # print(f"Lỗi lấy chi tiết bài: {e}")
        return None, None, None, None


# Hàm chính để cào dữ liệu
def scrape_data(base_url, target_count=300):
    data = []
    page = 1
    session = requests.Session()

    print(f"--- Bắt đầu cào dữ liệu từ: {base_url} ---")
    pbar = tqdm(total=target_count, desc="Tiến độ", unit="bài")

    # Dùng try-finally bao quanh vòng lặp lớn để đảm bảo luôn lưu dữ liệu
    try:
        while len(data) < target_count:
            # Tạo link phân trang
            url = f"{base_url}-p{page}" if page > 1 else base_url

            try:
                response = session.get(url, headers=HEADERS, timeout=10)
                if response.status_code != 200:
                    print(f"\nKhông truy cập được trang {page}. Bỏ qua.")
                    page += 1
                    continue

                soup = BeautifulSoup(response.content, "html.parser")
                articles = soup.find_all("h3", class_="title-news")

                if not articles:
                    print(f"\nKhông tìm thấy bài viết nào ở trang {page}. Dừng.")
                    break

                for article in articles:
                    if len(data) >= target_count: break  # Đủ số lượng thì thoát ngay

                    a_tag = article.find("a")
                    if not a_tag: continue

                    link = a_tag.get("href")
                    if "video" in link or "podcast" in link: continue

                    # Lấy chi tiết
                    date, title, sapo, body = get_article_content(session, link)

                    if title and body:
                        data.append({
                            "date": date,
                            "title": title,
                            "sapo": sapo,
                            "content_text": f"{title} {sapo} {body}",
                            "link": link
                        })
                        pbar.update(1)

                    # Nghỉ ngẫu nhiên nhẹ
                    time.sleep(random.uniform(0.2, 0.5))

                # Cứ sau mỗi trang (page) thì lưu tạm file 1 lần để chắc ăn
                if data:
                    pd.DataFrame(data).to_csv("dataset.csv", index=False, encoding="utf-8-sig")

                page += 1  # Tăng trang
                time.sleep(1)  # Nghỉ giữa các trang

            except Exception as e:
                print(f"\nLỗi tại trang {page}: {e}")
                page += 1
                continue

    except KeyboardInterrupt:
        print("\n!!! Người dùng dừng chương trình thủ công !!!")

    finally:
        # Code trong này LUÔN chạy dù chương trình thành công hay bị lỗi/dừng
        pbar.close()
        if data:
            df = pd.DataFrame(data)
            df.to_csv("dataset.csv", index=False, encoding="utf-8-sig")
            print(f"\n-> Đã lưu {len(data)} bài vào file 'dataset.csv'")
        else:
            print("\n-> Không có dữ liệu để lưu.")


if __name__ == "__main__":
    # Xóa file cũ nếu có để cào mới
    if os.path.exists("dataset.csv"):
        os.remove("dataset.csv")

    # Chạy hàm cào
    scrape_data("https://vnexpress.net/thoi-su", target_count=350)