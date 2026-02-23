from POP import PromptFunction
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
pf = PromptFunction(prompt="Give me 3 names for a <<<thing>>>.", client="openai")
result = pf.execute(thing="robot")

print(result)
print(pf.last_usage["source"])        # provider | estimate | hybrid | none
print(pf.last_usage["total_tokens"])  # canonical total used by POP
print(pf.get_usage_summary())         # cumulative counters