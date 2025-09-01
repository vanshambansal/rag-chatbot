import json
from newspaper import Article
from pathlib import Path

url = "https://example.com"

article = Article(url)
article.download()
article.parse()

text = article.text

output_file = Path("data/scraped_content.json")
output_file.parent.mkdir(exist_ok=True)
json.dump({"url": url, "text": text}, open(output_file, "w", encoding="utf-8"), indent=4)

print(f"Scraped {len(text)} characters from {url}")
output_file.parent.mkdir(exist_ok=True) 
json.dump({"url": url, "text": text}, open(output_file, "w", encoding="utf-8"), indent=4)

print(f"Scraped {len(text)} characters from {url}")
