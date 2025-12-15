import os
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ---------------- CONFIG ----------------
SEARCH_QUERY = "leather bag"  # Change to target product
LABEL = "authentic"           # Change to suspicious for counterfeit
NUM_PRODUCTS = 10
OUTPUT_FOLDER = f"data/images/{LABEL}"
# -----------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Setup WebDriver Service with WebDriver Manager
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Search Amazon
search_url = f"https://www.amazon.in/s?k={SEARCH_QUERY.replace(' ', '+')}"
driver.get(search_url)
time.sleep(3)

product_elements = driver.find_elements(By.CSS_SELECTOR, "img.s-image")
image_urls = [
    elem.get_attribute("src")
    for elem in product_elements
    if elem.get_attribute("src")
]

print(f"Found {len(image_urls)} images.")

# Download images
for idx, url in enumerate(image_urls[:NUM_PRODUCTS]):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(OUTPUT_FOLDER, f"{LABEL}_{idx+1}.jpg"), "wb") as f:
            f.write(response.content)
    time.sleep(0.5)

driver.quit()

print(f"Downloaded {len(image_urls[:NUM_PRODUCTS])} images to {OUTPUT_FOLDER}")
