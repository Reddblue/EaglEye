import requests
from transformers import BartTokenizer, BartForConditionalGeneration

NEWS_API_KEY = "64f9b14871064270a20b4457ffab8ecd"  # Replace with your actual NewsAPI key

def get_all_articles(query, page_size=100):
    base_url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": NEWS_API_KEY,
        "q": query,
        "language": "en",
        "pageSize": page_size,
    }
    
    all_articles = []
    page = 1
    while True:
        params["page"] = page
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if data["status"] == "ok" and data["articles"]:
            all_articles.extend(data["articles"])
            if len(data["articles"]) < page_size:
                # All articles retrieved, no more pages left
                break
            page += 1
        else:
            # No more articles or an error occurred
            break
    
    return all_articles

def generate_summary(article_text):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    inputs = tokenizer.encode("summarize: " + article_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def create_conclusion(article):
    # Manually create a conclusion based on the article's content (this is a simplified approach)
    conclusion = f"This article discusses {article['title']}. {article['description']} " \
                 f"It provides insights into the current financial landscape " \
                 f"and may have implications for the economy. The article emphasizes " \
                 f"the importance of monitoring such developments closely."
    return conclusion

def write_news_to_file(articles, filename):
    with open(filename, "w", encoding="utf-8") as file:
        for article in articles:
            article_text = article["content"] if "content" in article else article["description"]
            summary = generate_summary(article_text)
            conclusion = create_conclusion(article)
            file.write("Title: " + article["title"] + "\n")
            file.write("Summary: " + summary + "\n")
            file.write("Conclusion: " + conclusion + "\n")
            file.write("Source: " + article["source"]["name"] + "\n")
            file.write("URL: " + article["url"] + "\n")
            file.write("=" * 50 + "\n")

def main():
    query = "financial"  # Replace with your desired query
    articles = get_all_articles(query, page_size=100)

    output_filename = "financial_news_with_summaries.txt"
    write_news_to_file(articles, output_filename)
    print(f"All articles with summaries and conclusions written to {output_filename}")

if __name__ == "__main__":
    main()