import asyncio
from browser_use import Agent, Browser
from langchain_openai import ChatOpenAI

async def main():
    agent = Agent(
        task=(
            "1. Open https://colab.research.google.com\n"
            "2. Upload and open /Users/nev4rb14su/workspace/lora-trainer/examples/colab_lora_trainer.ipynb\n"
            "3. Click the 'Files' icon on the left to open the file explorer.\n"
            "4. Upload /Users/nev4rb14su/workspace/lora-trainer/examples/fern_new.zip to the /content folder.\n"
            "5. Run Cell 4 to Cell 8 in the notebook.\n"
            "6. Report progress."
        ),
        llm=ChatOpenAI(model="gpt-4o"),
    )
    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
