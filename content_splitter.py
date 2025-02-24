import pandas as pd
from POP import PromptFunction, load_prompt
import json

# Initialize the prompt function with the title extraction prompt.
pf = PromptFunction(load_prompt("prompts/corpus_splitter.md"))
fmt = json.loads('''
{
  "name": "Story_Title_Extraction",
  "schema": {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "titles": {
        "type": "array",
        "items": {"type": "string"},
        "description": "An array of story titles extracted from the text. If no titles are found, the array is empty."
      }
    },
    "required": ["titles"]
  }
}
''')

def prompt_function(text_chunk: str) -> dict:
    """
    Process a text chunk to extract story titles.
    Expected to return a dict with key 'titles' that is a list of strings.
    """
    result = pf.execute(
        text_chunk,
        sys=f"Here's what the json schema looks like: \n{fmt}"
    ).replace("```json\n", "").replace("```", "")
    print(result)
    return json.loads(result)

def chunk_text(text: str, chunk_size: int = 50, overlap: int = 200) -> list:
    """
    Breaks text into overlapping chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_sorted_title_positions(full_text: str, titles: list) -> list:
    """
    Find the positions of each title in the full text and return a sorted list of (position, title) tuples.
    """
    title_positions = []
    for title in titles:
        pos = full_text.find(title)
        if pos != -1:
            title_positions.append((pos, title))
    title_positions.sort(key=lambda x: x[0])
    
    # Deduplicate titles that are very close together.
    deduped = []
    last_pos = -100
    for pos, title in title_positions:
        if pos - last_pos > 10:
            deduped.append((pos, title))
            last_pos = pos
    return deduped

def extract_stories_from_title_positions(full_text: str, title_positions: list) -> list:
    """
    Given the full text and sorted title positions, extract story segments.
    Each story segment includes the title and its corresponding text.
    If no titles are found, the entire text is treated as one story with an empty title.
    """
    stories = []
    if not title_positions:
        stories.append({"title": "", "story": full_text.strip()})
        return stories
    
    for i, (pos, title) in enumerate(title_positions):
        start_index = pos
        if i < len(title_positions) - 1:
            end_index = title_positions[i+1][0]
        else:
            end_index = len(full_text)
        story_text = full_text[start_index:end_index].strip()
        stories.append({"title": title, "story": story_text})
    return stories

def process_large_text_file(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> pd.DataFrame:
    # Read the full text file.
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Chunk the text.
    text_chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
    
    all_titles = []
    
    # Process each chunk to extract titles.
    for i, chunk in enumerate(text_chunks):
        result = prompt_function(chunk)
        titles = result.get("titles", [])
        print(f"Chunk {i+1}/{len(text_chunks)}: Found {len(titles)} title(s).")
        all_titles.extend(titles)
    
    # Deduplicate titles while preserving order.
    unique_titles = list(set(all_titles))
    title_positions = get_sorted_title_positions(full_text, unique_titles)
    
    # Extract stories along with their titles.
    stories = extract_stories_from_title_positions(full_text, title_positions)
    
    # Create a DataFrame with columns "title" and "story".
    df = pd.DataFrame(stories)
    return df

# Example usage:
if __name__ == "__main__":
    #file_path = "test.txt"
    file_path = "cleaned_merged_fairy_tales_without_eos.txt"
    df_stories = process_large_text_file(file_path, chunk_size=200000, overlap=200)
    
    # Save the DataFrame to a CSV file.
    df_stories.to_csv("merged_stories.csv", index=False)
    
    print("Processing complete. Stories saved to merged_stories.csv.")
