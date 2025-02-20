import requests
from vanna_openai_connector import VannaOpenAIConnector

class NucleusVannaConnector(VannaOpenAIConnector):
    def __init__(self, api_key, nucleus_base_url="https://nucleus.yourcompany.com/api", *args, **kwargs):
        super().__init__(*args, **kwargs)  # Retain Vanna's initialization
        self.api_key = api_key
        self.nucleus_base_url = nucleus_base_url

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _call_nucleus_api(self, endpoint, payload):
        url = f"{self.nucleus_base_url}/{endpoint}"
        response = requests.post(url, json=payload, headers=self._get_headers())
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def complete_with_rag(self, prompt, documents=None, model="gpt-4", temperature=0.7, max_tokens=150):
        """
        This method extends the original Vanna connector to use your private 'nucleus' OpenAI layer for RAG.
        It will retrieve relevant documents and pass them to your nucleus model for completion.
        """
        # Step 1: Use RAG to retrieve documents based on the prompt
        documents = self.retrieve_documents_for_rag(prompt, documents)  # Assuming this is your RAG function

        # Step 2: Construct the payload to pass to Nucleus
        enriched_prompt = f"{prompt}\n\nRelevant Documents:\n" + "\n".join(documents)
        
        payload = {
            "model": model,
            "prompt": enriched_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Step 3: Call your private Nucleus API instead of OpenAI
        nucleus_response = self._call_nucleus_api("v1/completions", payload)
        
        # Step 4: Extract the response and return it
        return nucleus_response.get("choices", [{}])[0].get("text", "")

    def retrieve_documents_for_rag(self, prompt, documents=None):
        """
        Assuming you have a document retrieval function based on the RAG mechanism.
        This can be integrated with any document storage or search engine you are using.
        """
        # Retrieve the documents based on your RAG mechanism
        if documents is None:
            documents = ["Document 1 content...", "Document 2 content...", "Document 3 content..."]  # Example
        return documents


# v2
import requests
from vanna_openai_connector import VannaOpenAIConnector  # Import Vanna connector

class NucleusVannaConnector(VannaOpenAIConnector):
    def __init__(self, api_key, nucleus_url="https://nucleus.yourcompany.com/api", **kwargs):
        """
        Initialize the connector with the given API key and nucleus URL.
        """
        super().__init__(api_key=api_key, **kwargs)
        self.nucleus_url = nucleus_url
    
    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def complete_prompt(self, prompt, model="gpt-4", temperature=0.7, max_tokens=150):
        """
        Override the prompt completion method to use the nucleus API instead of OpenAI's API.
        """
        url = f"{self.nucleus_url}/v1/completions"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, json=payload, headers=self._get_headers())
        
        if response.status_code == 200:
            return response.json()["choices"][0]["text"]
        else:
            response.raise_for_status()

    def get_embeddings(self, text, model="text-embedding-ada-002"):
        """
        Override the embeddings method to use the nucleus API.
        """
        url = f"{self.nucleus_url}/v1/embeddings"
        
        payload = {
            "model": model,
            "input": text
        }
        
        response = requests.post(url, json=payload, headers=self._get_headers())
        
        if response.status_code == 200:
            return response.json()["data"]
        else:
            response.raise_for_status()

    def rag_query(self, query, doc_search_function, model="gpt-4", temperature=0.7, max_tokens=150):
        """
        Override the RAG functionality to use nucleus embeddings for information retrieval.
        The doc_search_function is expected to return relevant documents or information based on the query.
        """
        # Step 1: Use nucleus to get embeddings for the query
        query_embedding = self.get_embeddings(query)
        
        # Step 2: Use doc_search_function to retrieve relevant documents using query_embedding
        relevant_docs = doc_search_function(query_embedding)
        
        # Step 3: Construct a new prompt combining relevant docs and the original query
        combined_prompt = self._construct_rag_prompt(query, relevant_docs)
        
        # Step 4: Get a response using nucleus' completion API
        response = self.complete_prompt(combined_prompt, model, temperature, max_tokens)
        return response

    def _construct_rag_prompt(self, query, relevant_docs):
        """
        Helper method to create a RAG-style prompt with the query and relevant documents.
        """
        # You can customize how relevant_docs are formatted and added to the prompt
        doc_text = "\n".join([doc['text'] for doc in relevant_docs])  # Assuming docs contain a 'text' field
        return f"Given the following documents, answer the question:\n{doc_text}\n\nQuestion: {query}\nAnswer:"
