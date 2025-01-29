import dspy
import os
from dotenv import load_dotenv

load_dotenv()


lm = dspy.LM("azure_ai/gpt-4o-mini", api_key=os.getenv("AZURE_AI_API_KEY"), api_base=os.getenv("AZURE_AI_API_BASE"), api_version=os.getenv("AZURE_AI_API_VERSION"))
dspy.configure(lm=lm)

response = lm("What is capital of USA?", temperature=0.7) 
print(response)