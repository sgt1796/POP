```markdown
# Extracting Bedtime Stories and URLs from Webpage Text

## Goal
To accurately extract and organize bedtime story information from a given text version of a webpage, focusing on the story titles and their corresponding URLs, including the URL for the next page if available.

## Input Details
- **Source Material**: A text representation of a webpage containing:
  - Titles of bedtime stories.
  - The original URLs for these stories.
  - URLs to additional pages (e.g., "page 1", "page 2", or "next page"), if present.
- **Target Information**: The primary objective is to extract the titles of the stories and their direct URLs. Additionally, capture the URL for the next page when applicable.

## Extraction Guidelines
1. **Title and URL**:
   - Isolate and retrieve only the titles of stories and their respective URLs, excluding any site navigation elements, advertisements, or non-relevant content.
   - Trim any superfluous whitespace surrounding the titles or URLs.
   - Make sure ALL of the URL and titles except those in ads or recommendation fields are obtained.

2. **Guide line for locating next page**:
   - Identify and include the URL for the next page of stories, if such exists. Usually those are corresponding to a number (2,3,4,etc) or some text such as "next page". **Do not** choose from other source instead of leaving it blank.
   - **Don't return any urls that is not present in the text snapshot given you. **
   - Look for hint for the last page, such as: no next page, and there's no url for any page number larger than the current page. if so return empty str

3. **Handling Exceptions**:
   - Should the text lack identifiable story titles, return a specific error message: `"Refused: No suitable bedtime stories found."`
   - Bypass any story URLs that redirect incorrectly (e.g., back to the homepage) instead of to the actual story content.

4. **Output Presentation**:
   - Format the extracted data as a JSON object.
   - Use `titles_and_urls` as a key for an array of dictionaries, each containing a story's `title` and `url`.
   - Maintain clarity and conciseness in listing titles and URLs.
   - Include a `next_page` key with the URL to the next page of stories if available; otherwise, leave this field empty.
   - make sure the url to next page is actually links to next page. 

**Example Output for Valid Webpage Content**:

```json
{
  "titles_and_urls": [
    {
      "title": "Cinderella (Classic)",
      "url": "https://storiestogrowby.org/story/cinderella-fairy-tale-english-story-for-kids/"
    },
    {
      "title": "The Contest of the Fairies",
      "url": "https://storiestogrowby.org/story/rosanella-contest-fairies/"
    }
    // Additional stories here
  ],
  "next_page": ""
}
```

**Example Output for Invalid Webpage Content**:

```json
{
    "error": "Refused: No suitable bedtime stories found."
}
```
```
