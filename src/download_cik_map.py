import requests
import csv
import json

URL = "https://www.sec.gov/files/company_tickers.json"

def fetch_sec_tickers(url: str) -> dict:
    headers = {
        "User-Agent": "student research project (griffinhornbrook@gmail.com)",  # SEC requires a User-Agent header
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def save_to_csv(data: dict, output_file: str) -> None:
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "company_name", "cik"])

        for entry in data.values():
            writer.writerow([
                entry["ticker"],
                entry["title"],
                str(entry["cik_str"]).zfill(10),  # CIKs are zero-padded to 10 digits
            ])

def main():
    output_file = "ticker_cik_map.csv"

    print(f"Fetching data from {URL} ...")
    data = fetch_sec_tickers(URL)

    print(f"Saving {len(data)} records to {output_file} ...")
    save_to_csv(data, output_file)

    print(f"Done! Data saved to '{output_file}'.")

if __name__ == "__main__":
    main()