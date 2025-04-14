from agentpro import AgentPro
from agentpro.tools import AresInternetTool, CodeEngine, YouTubeSearchTool, SlideGenerationTool # ADD MORE TOOLS WHEN AVAILABLE
import os
import dotenv
def main():
    dotenv.load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the agent.")
        return
    if not os.environ.get("TRAVERSAAL_ARES_API_KEY"):
        print("Warning: TRAVERSAAL_ARES_API_KEY environment variable is not set.")
        print("AresInternetTool will not be available.")
        tools = [CodeEngine(), YouTubeSearchTool(), SlideGenerationTool()]
    else:
        tools = [AresInternetTool(), CodeEngine(), YouTubeSearchTool(), SlideGenerationTool()] # ADD MORE TOOLS WHEN AVAILABLE
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Warning: OPENROUTER_API_KEY environment variable is not set.")
        print("OpenRouter functionality may be limited.")
    if not os.environ.get("MODEL_NAME"):
        print("Warning: MODEL_NAME environment variable is not set.")
        print("Default model (GPT-4o-mini) will be used.")    
    agent = AgentPro(tools=tools)
    print("AgentPro is initialized and ready. Enter 'quit' to exit.")
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    while True:
        user_input = input("\nEnter your query: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        try:
            response = agent(user_input)
            print(f"\nAgent Response:\n{response}")
        except Exception as e:
            print(f"Error: {e}")
if __name__ == "__main__":
    main()
