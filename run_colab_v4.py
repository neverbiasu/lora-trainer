import asyncio
from browser_use import Agent
from langchain_openai import ChatOpenAI

async def main():
    agent = Agent(
        task=(
            "1. Open https://colab.research.google.com\n"
            "2. Upload and open /Users/nev4rb14su/workspace/lora-trainer/examples/colab_lora_trainer.ipynb\n"
            "3. Click the 'Files' icon on the left to open the file explorer.\n"
            "4. Upload /Users/nev4rb14su/workspace/lora-trainer/examples/fern_new.zip to the /content folder.\n"
            "5. Run cells 4 through 8 sequentially.\n"
            "6. Confirm the training has started."
        ),
        llm=ChatOpenAI(model="gpt-4o"),
    )
    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
