import openai
from phi.agent import Agent
import phi.api
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app

import os
phi.api=os.getenv("PHI_API_KEY")

websearch_agent = Agent(name="web search agent", 
                        role="search the web for information",
                        model=Groq(id="llama-3.3-70b-versatile", 
                                   api_key='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'),
                        tools=[DuckDuckGo()],
                        instructions=["Always include sources"],
                        show_tool_calls=True,
                        markdown=True,
)

finance_agent= Agent(name='Finance AI agent',
                     role='get financial data',
                     model=Groq(id="llama-3.3-70b-versatile",
                                api_key='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'),
                     tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, 
                            company_news=True)],
                     instructions=["Use tables to display the data"],
                     show_tool_calls=True,
                     markdown=True

)

app=Playground(agents=[finance_agent,websearch_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app", reload=True)

