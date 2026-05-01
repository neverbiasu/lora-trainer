import asyncio
from browser_use import Agent
from langchain_openai import ChatOpenAI

class LLMWrapper:
    def __init__(self, llm):
        self.llm = llm
        self.provider = "openai"
    def __getattr__(self, name):
        return getattr(self.llm, name)
    async def invoke(self, *args, **kwargs):
        return await self.llm.ainvoke(*args, **kwargs)
    def bind_tools(self, *args, **kwargs):
        return self.llm.bind_tools(*args, **kwargs)

async def main():
    llm = ChatOpenAI(model="gpt-4o")
    # Wrap LLM if needed, though browser-use should ideally handle it.
    # We will try passing the llm directly first, but the previous error showed it checks .provider.
    # Let's try adding the attribute directly to the object.
    setattr(llm, 'provider', 'openai')
    
    agent = Agent(
        task=(
            "1. Open https://colab.research.google.com\n"
            "2. Click 'Upload' tab and upload /Users/nev4rb14su/workspace/lora-trainer/examples/colab_lora_trainer.ipynb\n"
            "3. Click 'Files' icon on left.\n"
            "4. Upload /Users/nev4rb14su/workspace/lora-trainer/examples/fern_new.zip to /content.\n"
            "5. Run Cell 4 to Cell 8 sequentially.\n"
            "6. Done."
        ),
        llm=llm,
    )
    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
