from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from os import path
import requests
import hashlib
import time

TOTAL_PAGES = 570
col_index = ['name', 'quote_desc', 'quote', 'link', 'date_line', 'rating', 'sha256']
local_raw_data = "./politifact_data.csv"


def get_politifacts_page(page_num):
    page = requests.get(f"https://www.politifact.com/factchecks/list/?page={page_num}")
    soup = BeautifulSoup(page.content, 'html.parser')
    page_data = pd.DataFrame(columns=col_index)
    items = soup.findAll('article', class_="m-statement")
    for item in items:
        # Who (or what) is the quote attributed to
        name = item.find('a', class_="m-statement__name").get_text().strip()
        # Where was the quote made?
        quote_desc = item.find('div', class_="m-statement__desc").get_text().strip()
        # What is the quote
        quote = item.find('div', class_="m-statement__quote").find('a')
        link = f"https://politifact.com{quote['href'].strip()}"
        quote_text = quote.get_text().strip()
        # Date line with attribution, used for SHA-256 signature
        date_line = item.find('footer', class_="m-statement__footer").get_text().strip()
        # Rating - Label to be predicted. Might filter some of these out later
        rating = item.find('div', class_="m-statement__meter").find('picture').find('img')['alt']

        # SHA 256 hash used for identifying duplicate entries.
        # Will also be converted to an int for repeatable train/validation/test splits.
        sha256 = hashlib.sha256(f"{name}{quote_desc}{quote_text}{link}{date_line}{rating}".encode()).hexdigest()

        new_row = {col_index[0]: name, col_index[1]: quote_desc, col_index[2]: quote_text, col_index[3]: link,
                   col_index[4]: date_line, col_index[5]: rating, col_index[6]: sha256}
        page_data = page_data.append(new_row, ignore_index=True)
    return page_data


def update_politifact_data_set(end_page=1, start_page=1, data_set=None, wait_time=3):
    if data_set is None:
        data_set = pd.DataFrame(columns=col_index)

    if end_page > TOTAL_PAGES:
        end_page = TOTAL_PAGES

    for page_number in tqdm(range(end_page, start_page - 1, -1)):
        data_set = get_politifacts_page(page_num=page_number).append(data_set, ignore_index=True)
        # Delay between page requests. Don't be rude and slam their server.
        time.sleep(wait_time)

    # Remove any duplicate entries
    data_set = data_set.drop_duplicates(subset="sha256")

    return data_set


if __name__ == "__main__":
    if path.exists(local_raw_data):
        data = pd.read_csv(local_raw_data, sep='|')
    else:
        data = None

    data = update_politifact_data_set(5, data_set=data)

    data.to_csv(local_raw_data, header=True, index=False, sep='|')
