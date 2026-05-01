import asyncio
import os
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser

# Simplified approach: browser-use prefers its own LLM wrappers OR specific LangChain versions
# Let's try to upgrade everything or use a more robust monkeypatch

async def main():
    try:
        browser = Browser()
        
        # We'll use the simplest Agent call and hope the environment's LLM is fixed or use a direct one
        # If ChatOpenAI still fails, we might need a different LLM class from browser-use if exists
        
        agent = Agent(
            task="Go to https://colab.research.google.com/. Open the Files sidebar and upload /Users/nev4rb14su/Downloads/fern_new.zip to /content/.",
            llm=ChatOpenAI(model="gpt-4o"),
            browser=browser,
        )
        
        await agent.run()
    finally:
        await browser.close()

if __name__ == "__main__":
    # Ensure API Key is set in current environment
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY is not set.")
    else:
        asyncio.run(main())
