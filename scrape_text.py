import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ---------------- CONFIG ----------------
SEARCH_QUERY = "leather bag"  # Change to target product
LABEL = "authentic"           # Change to suspicious for counterfeit
NUM_PRODUCTS = 10
OUTPUT_FILE = "data/text/product_descriptions.csv"
# -----------------------------------------

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

data = []

# Setup WebDriver Service with WebDriver Manager
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Search Amazon
search_url = f"https://www.amazon.in/s?k={SEARCH_QUERY.replace(' ', '+')}"
driver.get(search_url)
time.sleep(3)

product_elements = driver.find_elements(By.CSS_SELECTOR, "a.a-link-normal.s-no-outline")
product_urls = []
for elem in product_elements:
    link = elem.get_attribute("href")
    if link and "amazon" in link:
        product_urls.append(link)
    if len(product_urls) >= NUM_PRODUCTS:
        break

print(f"Found {len(product_urls)} product URLs.")

for url in product_urls:
    driver.get(url)
    time.sleep(3)

    try:
        description = driver.find_element(By.ID, "productDescription").text
    except:
        description = ""

    try:
        reviews = driver.find_elements(By.CSS_SELECTOR, "span[data-hook='review-body']")
        reviews_text = " ".join([rev.text for rev in reviews[:3]])  # first 3 reviews
    except:
        reviews_text = ""

    full_text = description + " " + reviews_text
    if full_text.strip():
        data.append({"description": full_text, "label": LABEL})
    else:
        print(f"No data found for: {url}")

driver.quit()

df = pd.DataFrame(data)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Dataset saved to {OUTPUT_FILE}")
