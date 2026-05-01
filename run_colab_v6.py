import asyncio
from browser_use import Agent
from langchain_openai import ChatOpenAI
from pydantic import Field
from typing import Any

class WrappedChatOpenAI(ChatOpenAI):
    provider: str = Field(default="openai")

async def main():
    llm = WrappedChatOpenAI(model="gpt-4o")
    
    agent = Agent(
        task=(
            "1. Open https://colab.research.google.com\n"
            "2. Switch to 'Upload' tab and upload /Users/nev4rb14su/workspace/lora-trainer/examples/colab_lora_trainer.ipynb\n"
            "3. Click 'Files' on the left sidebar.\n"
            "4. Upload /Users/nev4rb14su/workspace/lora-trainer/examples/fern_new.zip to /content folder.\n"
            "5. Run cell 4 (unzip), cell 5, cell 6, cell 7, and cell 8 (training).\n"
            "6. Done."
        ),
        llm=llm,
    )
    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
