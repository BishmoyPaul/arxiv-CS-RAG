# ArXiv CS RAG

This repository hosts the code for [ArXiv CS RAG](https://huggingface.co/spaces/bishmoy/Arxiv-CS-RAG), a Huggingface space for searching paper embeddings and querying using large language models (LLMs) of your choice.  

## How The Space Works
https://github.com/user-attachments/assets/160a0fbd-3d59-49b1-9d4f-7535a7044e80

- **Input a Question**: The user inputs a question into the interface.
- **Abstract Retrieval**: The system uses ColBERTv2 to search ArXiv for the most relevant paper abstracts related to the question.
- **Contextual Answer Generation**: The retrieved abstracts are then fed into an LLM (Mistral or Gemma-based) to generate a detailed and accurate answer.
- **Output**: The final answer, along with the relevant abstracts, is displayed to the user.

## Key Features
- **Question-Based ArXiv Paper Retrieval**: Automatically fetches the most relevant ArXiv paper abstracts by using a question as input.
- **ColBERTv2 Retriever**: Employs ColBERTv2, a highly efficient retrieval model, to accurately find the most relevant abstracts based on the input question.
- **LLM-Powered Answers**: Uses advanced LLMs like Mistral or Gemma to generate comprehensive answers grounded in the retrieved paper abstracts.

## Codes
- [Create Embeddings from ArXiv Abstracts](https://github.com/BishmoyPaul/arxiv-CS-RAG/blob/main/generate_embeddings.ipynb)
- [Build Huggingface Space](https://huggingface.co/spaces/bishmoy/Arxiv-CS-RAG/tree/main)
