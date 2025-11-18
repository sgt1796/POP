```markdown
# Extracting Information and URLs from Webpage Text

## Goal
To accurately extract and organize relevant information from a given text version of a webpage, focusing on specific topic-related titles and their corresponding URLs, as well as the URL for the next page if available.

## Input Details
- **Source Material**: A text representation of a webpage containing:
  - Titles of content related to a specified topic.
  - The original URLs for these contents.
  - URLs to additional pages (e.g., "page 1", "page 2", or "next page"), if present.
- **Target Information**: The primary objective is to extract the titles of the content and their direct URLs. Additionally, capture the URL for the next page when applicable.

## Extraction Guidelines
1. **Title and URL**:
   - Isolate and retrieve only the titles of the content and their respective URLs, excluding any site navigation elements, advertisements, or non-relevant content.
   - Trim any superfluous whitespace surrounding the titles or URLs.
   - Ensure all relevant URLs and titles are obtained, except those in ads or recommendation fields.

2. **Guideline for Locating Next Page**:
   - Identify and include the URL for the next page of content, if such exists. Typically, these are associated with a number (2, 3, 4, etc.) or text such as "next page". **Do not** choose from other sources instead of leaving it blank.
   - **Don't return any URLs that are not present in the text snapshot provided to you.**
   - Look for hints indicating the last page, such as the absence of a "next page" link or a lack of URLs for any page number larger than the current page. If this is the case, return an empty string.

3. **Handling Exceptions**:
   - If the text lacks identifiable content titles, return a specific error message: `"Error: No suitable content found."`
   - Bypass any content URLs that redirect incorrectly (e.g., back to the homepage) instead of to the actual content.

4. **Output Presentation**:
   - Format the extracted data as a JSON object.
   - Use `titles_and_urls` as a key for an array of dictionaries, each containing a content's `title` and `url`.
   - Maintain clarity and conciseness in listing titles and URLs.
   - Include a `next_page` key with the URL to the next page of content if available; otherwise, leave this field empty.
   - Ensure the URL to the next page is indeed a link to the next page.

**Example Output for Valid Webpage Content**:

```json
{
  "titles_and_urls": [
    {
      "title": "Understanding Quantum Physics",
      "url": "https://example.com/quantum-physics/"
    },
    {
      "title": "The Basics of Machine Learning",
      "url": "https://example.com/machine-learning-basics/"
    }
    // Additional content here
  ],
  "next_page": ""
}
```

**Example Output for Invalid Webpage Content**:

```json
{
    "error": "Error: No suitable content found."
}
```
```