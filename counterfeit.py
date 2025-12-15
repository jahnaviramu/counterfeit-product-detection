import os
import time
import csv
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

# CONFIG
NUM_IMAGES = 10
NUM_TEXTS = 10
IMAGE_PATH = "data/images/counterfeit"
TEXT_PATH = "data/text"
KEYWORD = "cheap imitation leather bag"

# CREATE FOLDERS
os.makedirs(IMAGE_PATH, exist_ok=True)
os.makedirs(TEXT_PATH, exist_ok=True)

# SCRAPE IMAGES WITH SCROLLING
def scrape_images(keyword):
    print(f"Scraping {NUM_IMAGES} counterfeit images...")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    search_url = f"https://www.google.com/search?q={keyword}&tbm=isch"
    driver.get(search_url)
    time.sleep(2)

    # SCROLL DOWN to load more images
    for _ in range(5):
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    img_tags = soup.find_all("img")
    urls = [img.get("src") for img in img_tags if img.get("src")]

    downloaded = 0
    for i, url in enumerate(urls):
        if downloaded >= NUM_IMAGES:
            break
        if url.startswith("data:image"):  # skip base64
            continue
        try:
            img_data = requests.get(url, timeout=5).content
            with open(os.path.join(IMAGE_PATH, f"counterfeit_{i}.jpg"), "wb") as f:
                f.write(img_data)
            downloaded += 1
            print(f"Downloaded counterfeit_{i}.jpg")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    driver.quit()
    print(f"Downloaded {downloaded} counterfeit images.\n")


# SCRAPE TEXT DESCRIPTIONS
def scrape_text(keyword):
    print(f"Scraping {NUM_TEXTS} counterfeit text descriptions...")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    search_url = f"https://www.google.com/search?q={keyword}"
    driver.get(search_url)
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    descriptions = []
    desc_tags = soup.find_all("span")

    for tag in desc_tags:
        text = tag.get_text(strip=True)
        if len(text) > 30 and len(descriptions) < NUM_TEXTS:
            descriptions.append(text)

    driver.quit()
    return descriptions


# MAIN
if __name__ == "__main__":
    scrape_images(KEYWORD)

    output_file = os.path.join(TEXT_PATH, "counterfeit_descriptions.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["description", "label"])
        desc_list = scrape_text(KEYWORD)
        for desc in desc_list:
            writer.writerow([desc, "counterfeit"])

    print(f"Counterfeit descriptions saved to {output_file}")
    print("Counterfeit data collection complete!")
