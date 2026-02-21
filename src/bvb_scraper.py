import time
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


BASE_URL = "https://m.bvb.ro"
DETAILS_URL = "https://m.bvb.ro/FinancialInstruments/Details/FinancialInstrumentsDetails.aspx?s={ticker}"

TICKERS_FILE = "romanian_tickers.txt"
OUTPUT_CSV = "bvb_financial_reports_with_dates.csv"

def load_tickers(txt_file):
    with open(txt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def setup_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    return webdriver.Chrome(options=options)


def click_financial_info_tab(driver):
    tab = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.ID, "ctl00_body_TabsControlResponsive_4"))
    )
    
    driver.execute_script("arguments[0].scrollIntoView(true);", tab)
    driver.execute_script("arguments[0].click();", tab)

    # wait until reporting table exists
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.ID, "gvRepDoc"))
    )


def make_full_url(href):
    if href.startswith("http"):
        return href
    return BASE_URL + href


def extract_table_rows(driver):
    """
    Extract all rows in the gvRepDoc table:
    Each row contains:
      td[0] = filing_date
      td[2] contains <a href="...pdf|zip">
    """
    results = []

    table = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "gvRepDoc"))
    )

    tbody = table.find_element(By.TAG_NAME, "tbody")
    rows = tbody.find_elements(By.TAG_NAME, "tr")

    for row in rows:
        tds = row.find_elements(By.TAG_NAME, "td")
        if len(tds) < 3:
            continue

        filing_date = tds[0].text.strip()

        links = tds[2].find_elements(By.TAG_NAME, "a")
        for link in links:
            href = link.get_attribute("href")
            if not href:
                continue

            href_lower = href.lower()
            if href_lower.endswith(".pdf") or href_lower.endswith(".zip"):
                results.append({
                    "filing_date": filing_date,
                    "url": make_full_url(href)
                })

    return results


def scrape_ticker(driver, ticker):
    url = DETAILS_URL.format(ticker=ticker)
    driver.get(url)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
    except TimeoutException:
        print(f"[{ticker}] Page load timeout.")
        return []

    try:
        click_financial_info_tab(driver)
        time.sleep(2)
    except TimeoutException:
        print(f"[{ticker}] Could not find 'Informatii Financiare'.")
        return []

    all_results = []

    while True:
        try:
            # Extract current page rows
            all_results.extend(extract_table_rows(driver))

            # Check next button
            next_button = driver.find_element(By.ID, "gvRepDoc_next")

            if "disabled" in next_button.get_attribute("class"):
                break

            next_button.click()
            time.sleep(1.5)

        except StaleElementReferenceException:
            time.sleep(1)
            continue
        except Exception as e:
            print(f"[{ticker}] Error during scraping: {e}")
            break

    # Remove duplicates
    unique = {(r["filing_date"], r["url"]) for r in all_results}
    return [{"ticker": ticker, "filing_date": d, "url": u} for (d, u) in sorted(unique)]


def main():
    tickers = load_tickers(TICKERS_FILE)
    print(f"Loaded {len(tickers)} tickers.")

    driver = setup_driver(headless=True)

    all_data = []

    try:
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Scraping {ticker}...")

            rows = scrape_ticker(driver, ticker)

            print(f"[{ticker}] Found {len(rows)} documents.")

            all_data.extend(rows)

    finally:
        driver.quit()

    df = pd.DataFrame(all_data)

    # Sort output nicely
    if not df.empty:
        df = df.sort_values(["ticker", "filing_date", "url"]).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} total rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()