import pandas as pd
from nucleus_connector import NucleusVannaConnector

def rephrase_query(prompt: str, metadata: dict, api_key: str, model: str = "gpt-4-turbo", temperature: float = 0.2) -> str:
    """
    Rephrases a user question to be as clear as possible for generating better SQL queries, using metadata to replace slang, acronyms, etc.
    
    :param prompt: The natural language query to be rephrased.
    :param metadata: A dictionary containing metadata about the database schema.
    :param api_key: The API key for NucleusVannaConnector.
    :param model: The model to use for rephrasing.
    :param temperature: The temperature setting for response variability.
    :return: A rephrased query as a string.
    """    
    system_prompt = f"""
    You are an expert in refining user queries for optimal SQL generation. Given a user's natural language question and the following metadata, rephrase it to be as clear, structured, and unambiguous as possible while retaining its original intent.
    
    """
    
    # Initialize NucleusVannaConnector
    nucleus_connector = NucleusVannaConnector(api_key=api_key)
    
    # Call Nucleus API
    response = nucleus_connector.complete_prompt(system_prompt + "\n\n" + prompt, model=model, temperature=temperature)
    
    return response