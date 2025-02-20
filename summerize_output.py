import pandas as pd
from nucleus_connector import NucleusVannaConnector

def summarize_output(df: pd.DataFrame, api_key: str, model: str = "gpt-4"):
    # Convert DataFrame to dictionary
    sample_data = df.head(10).to_dict(orient='records')
    
    # Create prompt
    prompt = f"""
    Here is a sample of a dataset:
    {sample_data}
    
    Provide a summary of the data, including key insights, trends, and notable values. The response should always include a plain text summary, and if needed, generate Markdown-formatted table(s) for additional clarity.
    """
    
    # Initialize NucleusVannaConnector
    nucleus_connector = NucleusVannaConnector(api_key=api_key)
    
    # Call Nucleus API
    summary_text = nucleus_connector.complete_prompt(prompt, model=model)
    
    return summary_text