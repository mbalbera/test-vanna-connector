import pandas as pd
from nucleus_connector import NucleusVannaConnector

def generate_sql(prompt: str, api_key: str, model: str = "gpt-4-turbo", temperature: float = 0.2) -> str:
    """
    Generates an SQL query using NucleusVannaConnector, incorporating metadata for table and schema information.
    
    :param prompt: The natural language query describing the SQL query needed.
    :param metadata: A dictionary containing metadata about the database schema.
    :param api_key: The API key for NucleusVannaConnector.
    :param model: The model to use for generating the SQL query.
    :param temperature: The temperature setting for response variability.
    :return: A generated SQL query as a string.
    """

    system_prompt = f"""
    You are an expert SQL query generator. Given a natural language description and the following metadata, you will generate an optimized and correct SQL query.
    
    Ensure the SQL is syntactically valid and follows best practices.
    """
    
    # Initialize NucleusVannaConnector
    nucleus_connector = NucleusVannaConnector(api_key=api_key)
    
    # Call Nucleus API
    response = nucleus_connector.complete_prompt(system_prompt + "\n\n" + prompt, model=model, temperature=temperature)
    
    return response