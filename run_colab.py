import asyncio
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI

async def main():
    browser = Browser(
        config=BrowserConfig(
            headless=False,
        )
    )
    agent = Agent(
        task=(
            "1. Open https://colab.research.google.com\n"
            "2. Upload and open the notebook /Users/nev4rb14su/workspace/lora-trainer/examples/colab_lora_trainer.ipynb\n"
            "3. Upload the file /Users/nev4rb14su/workspace/lora-trainer/examples/fern_new.zip to the /content directory in Colab using the Files side panel.\n"
            "4. Run cell 4 (Extract dataset) and subsequent cells up to cell 8 (Training).\n"
            "5. If you encounter a file upload dialog that you cannot handle, please report the UI state."
        ),
        llm=ChatOpenAI(model="gpt-4o"),
        browser=browser,
    )
    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
