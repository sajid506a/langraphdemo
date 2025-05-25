import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults
import os

# Load environment variables
load_dotenv()

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
)

# Initialize search tool
search_tool = TavilySearchResults(search_developer_key=os.getenv("TAVILY_API_KEY"))

def main():
    prompt = "how is weather in hyderabad in degrees celsius"
    
    agent = initialize_agent(
        tools=[search_tool],  
        llm=llm,
        agent_type="zero-shot-react-description",
        verbose=True,
    )
    
    response = agent.invoke(prompt)
    print(response)

if __name__ == "__main__":
    main()