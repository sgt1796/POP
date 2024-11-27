import GPT_crawler_utils as GPT_crawler
from openai import OpenAI
from dotenv import load_dotenv
import json
import argparse
import csv

GPT_MAX_TOKENS = 30000
load_dotenv()
client = OpenAI()

## Chunks and merge the website snapshot
excluded_selectors = [
    # Advertisements
    ".ads", ".ad", ".advertisement", ".sponsored", ".promo", ".banner-ad", ".google-ad", ".ad-container",
    # Sidebar and widgets
    ".sidebar", ".widget", ".widget-container", ".sidebar-content", ".related-posts", ".recommendations",
    ".tag-cloud", ".sidebar-widget", ".right-sidebar", ".left-sidebar",
    # Popups and modals
    ".modal", ".popup", ".overlay", ".pop-up", ".cookie-banner", ".consent-banner", ".alert",
    # Social media and sharing
    ".social", ".share", ".share-buttons", ".follow-us", ".social-links", ".social-widget", ".social-icons",
    # Other common non-content elements
    ".search", ".search-bar", ".search-box", ".contact-form", ".newsletter", ".subscribe", ".logo",
    ".branding", ".terms", ".privacy-policy", ".faq", ".site-map"
]

def get_category_urls(url):
    snapshot = GPT_crawler.get_text_snapshot(url, exclude_selector=excluded_selectors, links_at_end=True) # put links at the end for better crawling
    snapshot_links = snapshot[snapshot.find("Links/Buttons:"):]

    completion_category = GPT_crawler.find_available_content(snapshot_links)
    categories = json.loads(completion_category.choices[0].message.content).get('categories', {})

    ### check correctness of the categories url
    for category, category_url in categories.copy().items():
        if GPT_crawler.GPT_boolean(category_url, "Does this website contain > 10 stories (with title and urls)?"):
            print(f"The category '{category}' contains stories for kids.")
        else:
            print(f"The category '{category}' does not contain stories for kids.")
            del categories[category] 
    print()
    return categories

def get_titles_and_urls(categories: dict):
    '''
    Get the titles and urls of the stories of each category in the categories dictionary.
    
    Args:
        categories (dict): A dictionary of categories and their urls, in the form {"category": "url"}. Should be the output of get_category_urls.
    '''
    titles_and_urls = {}  
    send_full_page = False # whether send full page to the model
    for category, category_url in categories.items():
        print(f"Category: {category}\nURL: {category_url}\n")
        page = 0
        next_page = category_url
        cached_snapshot = None  # Cache the valid snapshot

        while next_page:
            page += 1
            print(f"[Page: {page}]")

            # Fetch or reuse snapshot
            if cached_snapshot and cached_snapshot.get("url") == next_page:
                snapshot = cached_snapshot["snapshot"]
            else:
                snapshot = f"{GPT_crawler.get_text_snapshot(next_page, exclude_selector=excluded_selectors, links_at_end=True)}\nCurrent Page Number: {page}" # This might help the model to stop at the end of the page
                cached_snapshot = None  # Reset cache

            snapshot_links = snapshot[snapshot.find("Links/Buttons:"):]

            if not send_full_page:
                completion_titles = GPT_crawler.get_titles_and_urls(snapshot_links)
            else:
                completion_titles = GPT_crawler.get_titles_and_urls(snapshot)

            titles_and_urls_list = json.loads(completion_titles.choices[0].message.content).get('titles_and_urls', [])
            
            ## if the crawled content is empty, try to crawl the full page
            if not titles_and_urls_list and not send_full_page:
                print("Empty content. Setting send_full_page to True and retrying...")
                send_full_page = True
                page -= 1
                continue # restart the loop with full page mode

            for title_and_url in titles_and_urls_list:
                title = title_and_url.get('title', "")
                url = title_and_url.get('url', "")
                titles_and_urls[title] = (url, category)
                print(f"{title}: {titles_and_urls[title]}")

            next_page = json.loads(completion_titles.choices[0].message.content).get('next_page', "")
            ## validate the next page exists
            query = f"the current page is {page}, does this page have a link to next page or page {page+1}?"
            snapshot = f"{GPT_crawler.get_text_snapshot(next_page, exclude_selector=excluded_selectors, links_at_end=True)}\nCurrent Page Number: {page}" # This might help the model to stop at the end of the page
            if GPT_crawler.GPT_boolean(snapshot, query):
                cached_snapshot = {"url": next_page, "snapshot": snapshot}
            else:
                print(f"Next page does not exist.")
                next_page = ""

            print(f"Next page: {next_page}")
        return titles_and_urls

def get_content(titles_and_urls):
    stories = {}

    for title, (url, category) in titles_and_urls.items():
        print(f"Title: {title}")
        print(f"Category: {category}")
        print(f"URL: {url}")
        print("")
        completion = GPT_crawler.get_content(url)
        try:
            content = json.loads(completion.choices[0].message.content).get('content', '')
            author = json.loads(completion.choices[0].message.content).get('author', '')
            next_page = json.loads(completion.choices[0].message.content).get('next_page', "") # should be empty
        except Exception as e:
            print(f"Error while fetching story content: {e}")
            print(completion)
            content = ''
            author = ''
            next_page = "" # currently not support multi-page story

        stories[title] = {
            'title': title,
            'author': author,
            'content': content,
            'category': category,
            'url': url
        }

    return stories

def main(args):
    ## Step 1
    ## Get the categories and their urls
    categories = get_category_urls(args.input) 
    for category, category_url in categories.items():
        print(category)
        print(category_url)

    ## Step 2
    ## Get the titles and urls of the stories of each category
    print("\nFetching titles and urls of the stories of each category...\n")
    titles_and_urls = get_titles_and_urls(categories) 

    ## Step 3
    ## Get the content of the stories
    stories = get_content(titles_and_urls) 
    # Transform the stories dictionary into a list of dictionaries
    stories_list = [
        {
            'title': title,
            'author': details.get('author', ''),
            'content': details.get('content', ''),
            'category': details.get('category', ''),
            'url': details.get('url', '')
        }
        for title, details in stories.items()
    ]

    with open(args.output, 'w', newline='') as file:
        fieldnames = ['title', 'author', 'content', 'category', 'url']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stories_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crawl stories from a website')
    parser.add_argument('-i', '--input', type=str, help='Input an url of the website to be crawled.')
    parser.add_argument('-o', '--output', type=str, help='Output file to save the crawled stories.')
    args = parser.parse_args()

    main(args)
