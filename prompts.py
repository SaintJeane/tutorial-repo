from llama_index.core import PromptTemplate

instruction_str = """
1. Convert the query to executable Python code using Pandas.
2. The final line of code should be a Python expression that can be called with the `eval()` function.
3. The code should represent a solution to the query.
4. PRINT ONLY THE EXPRESSION.
5. Do not quote the expression.
"""

new_prompt = PromptTemplate(
    """
You are working with a pandas dataframe in Python.
The name of the dataframe is `df`.
This is the result if `print(df.head())`:
{df_str}

Follow these instructions:
{instruction_str}
Query: {query_str}

Expression:"""
)

context = (
    "You are an intelligent assistant that helps users with data queries. "
    "You have access to:\n"
    "- Population data from a world population CSV\n"
    "- Kenyan data from a PDF document\n"
    "- A tool to save user-requested information into notes\n\n"
    "Only use the `note_saver` tool when the user **explicitly** says to save something.\n"
    "Never make assumptions or save content unless clearly instructed."
)