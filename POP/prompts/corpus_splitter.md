## Task: Extract Story Titles from a Large Corpus of Text Stories

You are provided with a segment of text that may contain one or more story titles. Your task is to identify and extract the title of each story present in the text.

### Guidelines:

1. **Identify Story Titles**: Look for clear indicators of story titles such as distinct headings, chapter titles, or any other formatting markers that signify the beginning of a new story.
2. **Single or Multiple Titles**: If the text contains a single story or if the title is missing because the story is truncated, return an empty list. If there are multiple story titles, extract each one.
3. **Preserve Original Formatting**: Extract each title exactly as it appears in the text. Do not modify punctuation, capitalization, or spacing.
4. **No Additional Text**: Return only the story titles without any surrounding text or commentary.
5. **Handle Edge Cases**: If no clear story title is found in the text, return an empty list.
6. If there seems to be no title, just return nothing. 
7. The chapters are NOT titles, do not include them as they can appear more than once.

### Return Format:

Return a JSON object with one property "titles", which is an array of strings. For example, if the text chunk includes the titles "The Selfish Giant" and "The Devoted Friend", then your output should be:

{
  "titles": ["The Selfish Giant", "The Devoted Friend"]
}

Remember:
- Do not include any markdown formatting, code blocks, or the "```" characters in your answer.
- All property names must be enclosed in double quotes.
- Your answer should be nothing but a JSON string that strictly follows the format described above.

Please process the text accordingly and return the result.