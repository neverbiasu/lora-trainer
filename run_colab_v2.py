import asyncio
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
import os

async def main():
    # Attempting to use the already initialized browser if possible, 
    # but browser-use usually starts its own instance.
    browser = Browser(
        config=BrowserConfig(
            headless=False,
        )
    )
    agent = Agent(
        task=(
            "1. Go to https://colab.research.google.com\n"
            "2. Upload the notebook from /Users/nev4rb14su/workspace/lora-trainer/examples/colab_lora_trainer.ipynb (File > Upload notebook)\n"
            "3. Once the notebook is open, click on the 'Files' icon on the left sidebar.\n"
            "4. Upload /Users/nev4rb14su/workspace/lora-trainer/examples/fern_new.zip to the /content directory.\n"
            "5. Run Cell 4 (Extraction), Cell 5, Cell 6, Cell 7, and Cell 8 (Training) in order.\n"
            "6. Confirm the training starts."
        ),
        llm=ChatOpenAI(model="gpt-4o"),
        browser=browser,
    )
    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
