## Bedtime Story Outline Generator (JSON Structured Output)

**Objective:**  
Generate a comprehensive and imaginative outline for a bedtime story that can later be expanded into full chapters. The output must be in valid JSON format following the structure provided below.

---

### User's Request:
<<<user_request>>>

---

### Instructions:

Please generate the bedtime story outline using the JSON structure detailed below:

1. **Story Overview** (`storyOverview` object):
   - **title**: Provide a creative and engaging title for the story.
   - **summary**: Write a brief summary capturing the main plot, setting, and overall mood of the story.

2. **Main Character Details** (`mainCharacter` object):
   - **name**: Name of the main character.
   - **background**: A short background story.
   - **personality**: Description of personality traits, motivations, strengths, and vulnerabilities.
   - **appearanceAndCostumes**: Details of physical appearance and distinctive clothing or costumes (include colors, styles, and any magical or unique elements).

3. **Setting Description** (`setting` object):
   - **primarySetting**: Describe the main setting(s) where the story takes place (e.g., a magical forest, a cozy village, a dreamy castle).
   - **atmosphereDetails**: Include sensory details (sights, sounds, smells) to bring the setting to life and fit the bedtime story theme.

4. **Chapter-by-Chapter Breakdown** (`chapters` array):
   Each chapter should be represented as an object with the following keys:
   - **chapterTitle**: A catchy title for the chapter.
   - **chapterSummary**: A brief overview of what happens in the chapter.
   - **keyEvents**: An array listing the main events, actions, or conflicts in the chapter.
   - **characterDevelopment**: Description of any new insights, challenges, or growth for the main character.
   - **supportingElements**: Additional characters, magical elements, or important items introduced in the chapter.
   - **chapterConclusion**: Explain how the chapter ends and sets up the next chapter.

5. **Supporting Characters & Antagonists** (`supportingCharacters` array):1
   Each supporting character or antagonist should be an object with:
   - **name**: Name of the character.
   - **description**: A brief description.
   - **costumesAndTraits**: Details about their appearance, costumes, and defining characteristics.

6. **Themes and Moral Lessons** (`themes` object):
   - **coreThemes**: An array of the central themes (e.g., friendship, bravery, kindness).
   - **lessons**: Outline any moral or educational lessons that the story will convey.

7. **Additional Elements** (`additionalElements` object):
   - **recurringMotifs**: Any symbols or motifs (like a magic key, a sparkling star) that recur throughout the story.
   - **specialFeatures**: Whimsical or fantastical elements (such as talking animals, enchanted objects) that add to the bedtime story magic.

---

### Output Requirements:
- **Format:** The output must be valid JSON.
- **Structure:** Follow the JSON keys and structure as specified above.
- **Tone:** The tone should be soothing, imaginative, and appropriate for a bedtime story.
- **Detail:** Each section must be detailed enough to provide a solid foundation for writing complete chapters later.

---

### Example JSON Output:

```json
{
  "storyOverview": {
    "title": "The Moonlit Adventures of Luna",
    "summary": "Luna, a curious and brave girl with a magical cloak, embarks on nightly adventures in an enchanted forest filled with talking animals and hidden secrets."
  },
  "mainCharacter": {
    "name": "Luna",
    "background": "A kind-hearted child from a small village, fascinated by the mysteries of the night.",
    "personality": "Curious, brave, and compassionate, with a gentle sense of humor.",
    "appearanceAndCostumes": "Wears a shimmering moonlit cloak, star-patterned pajamas, and carries a lantern that glows with magical light."
  },
  "setting": {
    "primarySetting": "An enchanted forest filled with ancient trees, sparkling streams, and hidden pathways.",
    "atmosphereDetails": "The forest hums with soft whispers, rustling leaves, and distant calls of nocturnal creatures, creating a magical ambiance."
  },
  "chapters": [
    {
      "chapterTitle": "The Whispering Woods",
      "chapterSummary": "Luna enters the enchanted forest and encounters mysterious whispers that lead her deeper into adventure.",
      "keyEvents": [
        "Luna hears soft, mysterious whispers as she steps into the forest.",
        "She meets a wise owl who offers cryptic clues.",
        "Faces a gentle challenge that teaches her to trust her instincts."
      ],
      "characterDevelopment": "Luna learns to trust her inner voice and the guidance of nature.",
      "supportingElements": "A wise owl, a glowing map, and a shimmering trail.",
      "chapterConclusion": "The chapter ends with Luna discovering a sparkling path that hints at further adventures.",
      "currentChapter": 1
    }
  ],
  "total_chapters": 1,
  "supportingCharacters": [
    {
      "name": "Oliver the Owl",
      "description": "A wise and playful owl who offers guidance on Luna's journey.",
      "costumesAndTraits": "Wears a tiny pair of spectacles and has feathers that shimmer in the moonlight."
    }
  ],
  "themes": {
    "coreThemes": ["Friendship", "Courage", "Curiosity"],
    "lessons": "The story emphasizes the importance of trust, the beauty of nature, and the magic in every adventure."
  },
  "additionalElements": {
    "recurringMotifs": "A sparkling star that appears at key moments in Luna's journey.",
    "specialFeatures": "Enchanted objects like a glowing lantern and a mysterious, magical map."
  }
}
