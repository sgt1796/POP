from POP import PromptFunction, get_text_snapshot
import argparse
import json
import csv

pf_load = PromptFunction()
def load_prompt(file_path):
    return pf_load.load_prompt(file_path)
content_finder = PromptFunction(load_prompt("prompts/content_finder.md"))
get_title_and_url = PromptFunction(load_prompt("prompts/get_title_and_url.md"))
get_content = PromptFunction(load_prompt("prompts/get_content.md"))

content_finder.set_temperature(0.1)  # default is 0.7
get_title_and_url.set_temperature(0.0)
get_content.set_temperature(0.0)

category_response_format = {
    "name": "article_categories",
    "schema": {
        "type": "object",
        "properties": {
            "categories": {
                "type": "object",
                "additionalProperties": {
                    "type": "string"
                }
            }
        },
        "additionalProperties": False,
        "required": ["categories"],
    }
}
title_response_format = {
    "name": "titles_and_urls",
    "schema": {
        "type": "object",
        "properties": {
            "titles_and_urls": {
                "description": "Titles and urls of the articles.",
                "type": "object",
                "additionalProperties": {
                    "type": "string"
                },
            },
            "next_page": {
                "description": "The URL of the next page to fetch titles and URLs from. Empty means no more pages.",
                "type": "string"
            },
        },
        "additionalProperties": False,
        "required": ["titles_and_urls", "next_page"],
    }
}
content_response_format = {
    "name": "contents",
    "schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string"
            }
        },
        "additionalProperties": False,
        "required": ["content"],
    }
}

def main(args):
    url = args.input
    user_request = args.request
    output_file = args.output
    cookie = args.cookie

    # Use the cookie value when calling get_text_snapshot via the helper.
    def get_snapshot(url, links_at_end=False):
        return get_text_snapshot(url, links_at_end=links_at_end, cookie=cookie)

    categories = content_finder.execute(get_snapshot(url, links_at_end=True),
                                          model="gpt-4o",
                                          ADD_BEFORE=user_request,
                                          fmt=category_response_format)
    categories = json.loads(categories).get("categories", {})

    titles_and_urls = {}
    error_log = []
    send_full_snapshot = False
    for category, category_url in categories.items():
        print(f"{category}: {category_url}")
        next_page = category_url
        page = 0

        while next_page:
            page += 1
            print(f"[Page: {page}]")
            snapshot = get_snapshot(next_page, links_at_end=True)
            snapshot_links = snapshot[snapshot.find("Links/Buttons:"):]
            titles = get_title_and_url.execute(snapshot_links,
                                               ADD_BEFORE=f"I'm looking for articles on {category}, please provide me with the title and URL of the articles.",
                                               model="gpt-4o-mini",
                                               fmt=title_response_format)
            try:
                parsed = json.loads(titles)
                next_page = parsed.get("next_page", "")
                titles_list = parsed.get("titles_and_urls", [])
            except Exception as e:
                next_page = ""
                titles_list = []
                error_log.append({"step": "title_and_url", "message": str(e), "response": titles})

            if not titles_list and not send_full_snapshot:
                print("Empty content. Setting send_full_snapshot to True and retrying...")
                send_full_snapshot = True
                page -= 1
                continue

            for title_and_url in titles_list:
                title = title_and_url.get('title', "")
                url = title_and_url.get('url', "")
                titles_and_urls[title] = (url, category)
                print(f"{title}: {titles_and_urls[title]}")

    contents = {}
    for title, (url, category) in titles_and_urls.items():
        print(f"Title: {title}")
        print(f"Category: {category}")
        print(f"URL: {url}")

        next_page = url
        content = ""
        author = ""
        while next_page:
            response = get_content.execute(next_page,
                                           ADD_BEFORE=f"I'm looking for the content of the article '{title}'.",
                                           model="gpt-4o-mini",
                                           fmt=content_response_format)
            try:
                parsed = json.loads(response)
                next_page = parsed.get("next_page", "")
                if not author:
                    author = parsed.get("author", "")
                content += parsed.get("content", "")
            except Exception as e:
                next_page = ""
                author = ""
                content = ""
                error_log.append({"step": "content", "message": str(e), "response": response})
        contents[title] = {"title": title,
                           "category": category,
                           "author": author,
                           "content": content,
                           "url": url}

    contents_list = [
        {
            'title': title,
            'author': details.get('author', ''),
            'content': details.get('content', ''),
            'category': details.get('category', ''),
            'url': details.get('url', '')
        }
        for title, details in contents.items()
    ]
    print("crawling complete!")
    with open(output_file, 'w', newline='') as file:
        print("saving data...")
        fieldnames = ['title', 'author', 'content', 'category', 'url']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(contents_list)

    with open("error_log.txt", "w") as file:
        print("saving error logs...")
        for error in error_log:
            file.write(json.dumps(error) + "\n")
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crawl stories from a website')
    parser.add_argument('-i', '--input', type=str, help='Input URL of the website to be crawled.')
    parser.add_argument('-o', '--output', type=str, help='Output file to save the crawled stories.')
    parser.add_argument('-r', '--request', type=str, help='User request to be added before the prompt.')
    parser.add_argument('--cookie', type=str, help='Cookie header value to use for authenticated crawling')
    args = parser.parse_args()

    main(args)
