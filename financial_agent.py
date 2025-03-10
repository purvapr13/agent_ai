from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

import openai

import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key="get your openai api key from env file here"

websearch_agent = Agent(name="web search agent", 
                        role="search the web for information",
                        model=Groq(id="llama-3.3-70b-versatile", 
                                   api_key='<your groq api key>'),
                        tools=[DuckDuckGo()],
                        instructions=["Always include sources"],
                        show_tool_calls=True,
                        markdown=True,
)

finance_agent= Agent(name='Finance AI agent',
                     role='get financial data',
                     model=Groq(id="llama-3.3-70b-versatile",
                                api_key='<your groq api key>'),
                     tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, 
                            company_news=True)],
                     instructions=["Use tables to display the data"],
                     show_tool_calls=True,
                     markdown=True

)

multi_ai_agent= Agent(
    team=[websearch_agent, finance_agent],
    model=Groq(id="llama-3.3-70b-versatile", api_key='<your groq api key>'),
    instructions=["Always include sources", "Use table to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("summarize analyst recommendation and share the latest news for NVIDIA", stream=True)
