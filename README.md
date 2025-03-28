# RAG Demo with LangChain, ChromaDB, and OpenAI

This is a basic JavaScript implementation of a Retrieval-Augmented Generation (RAG) system using:

- LangChain.js for orchestration
- ChromaDB for vector storage
- OpenAI API for embeddings and completions

## Project Structure

- `index.js` - Main application entry point
- `documents/` - Sample documents for ingestion
- `utils/` - Helper functions
- `.env` - Environment variables (OpenAI API key)

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Make sure your `.env` file contains your OpenAI API key:
   ```
   OPENAI_API_KEY=your-key-here
   ```

3. Run the application:
   ```bash
   npm start
   ```

## How It Works

1. Documents are loaded and split into chunks
2. OpenAI embeddings are generated for each chunk
3. Embeddings are stored in ChromaDB vector store
4. User query is processed through the same embedding model
5. Similar context is retrieved from the vector store
6. Retrieved context and query are sent to OpenAI for a complete response
