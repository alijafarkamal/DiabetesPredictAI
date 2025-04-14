from agentpro import AgentPro
from agentpro.tools import AresInternetTool, CodeEngine, YouTubeSearchTool, SlideGenerationTool
import os

def main():
    # Initialize tools
    try:
        ares_tool = AresInternetTool()
        code_tool = CodeEngine()
        youtube_tool = YouTubeSearchTool()
        slide_tool = SlideGenerationTool()
        
        # Create agent with tools
        agent = AgentPro(tools=[ares_tool, code_tool, youtube_tool, slide_tool])
        
        # Example tasks
        tasks = [
            "Generate a presentation deck on Supervised Fine-tuning",
            "Generate a chart comparing Nvidia stock to Google. Save the graph as comparison.png file. Execute the code using code engine",
            "Make me a diet plan by searching YouTube videos about keto diet"
        ]
        
        for i, task in enumerate(tasks):
            print(f"\n\n=== Running Example {i+1}: {task} ===\n")
            response = agent(task)
            print(f"\nFinal Answer: {response}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've set the required API keys as environment variables:")
        print("- OPENAI_API_KEY")
        print("- TRAVERSAAL_ARES_API_KEY")

if __name__ == "__main__":
    main()