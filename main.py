# from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
# from llama_index.llms import openai
from llama_index.llms.ollama import Ollama

from pdf import kenya_engine


# load_dotenv()

#Load the population csv
population_path = os.path.join("data", "world_population.csv")
population_df = pd.read_csv(population_path)

# print(population_df.head())

#setting up pandas query engine
population_query_engine = PandasQueryEngine(df=population_df, verbose=True, instruction_str=instruction_str)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})
# population_query_engine.query("what is the population of Kenya?")

#defining the tools for the model
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine, 
        metadata=ToolMetadata(
            name="population_data",
            description="Gives the information at the world population and demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=kenya_engine, 
        metadata=ToolMetadata(
            name="kenya_data",
            description="Gives the information about the country Kenya",
        ),
    ),

]

# llm = openai(model = "gpt-3.5-turbo-0613")

#setting up llm with Ollama using mistral model
llm = Ollama(model="mistral")

# initialize the ReAct Agent
agent = ReActAgent.from_tools(llm, tools=tools, verbose=True, context=context) #tell the agent to pick which tool which is best for the job
#agent = ReActAgent.from_tools(tools, llm = llm, verbose=True)

#command loop
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)
