### Task: Generate an OpenAI Function JSON Schema

You are given a natural language description of a function. Your task is to return a **valid JSON object** that defines an OpenAI-compatible function schema.

#### Requirements

1. **Output JSON only**: Do not include markdown or extra text.
2. **Top-level fields**: The JSON object must include both `"name"` and `"schema"`.
3. **Schema format**: Use JSON Schema Draft-07. The `"schema"` field must be an object with at least:
   - `"$schema": "http://json-schema.org/draft-07/schema#"`
   - `"type": "object"`
   - `"properties": { ... }`
   - `"required": [ ... ]` (if needed)
4. **Be precise**: Reflect the description accurately using types, constraints, and clear property names.

The user description will be provided in the user message. Return only the JSON object.
