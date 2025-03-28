// RAG Demo with LangChain, MemoryVectorStore, and OpenAI
import { config } from 'dotenv';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from '@langchain/openai';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { RunnableSequence } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { formatDocumentsAsString } from 'langchain/util/document';

// Load environment variables
config();

// Verify we have an OpenAI API key
if (!process.env.OPENAI_API_KEY) {
  console.error('Error: OPENAI_API_KEY is not set in the environment variables.');
  console.error('Please make sure you have a .env file with your OpenAI API key.');
  process.exit(1);
}

// Initialize models
const embeddings = new OpenAIEmbeddings();
const llm = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temperature: 0.2
});

// Main function to demonstrate RAG
async function runRAGDemo() {
  console.log('🚀 Starting RAG Demo with LangChain and OpenAI');
  
  // Step 1: Load documents
  console.log('\n📚 Loading documents...');
  const loader = new DirectoryLoader('./documents', {
    '.txt': (path) => new TextLoader(path)
  });
  
  const docs = await loader.load();
  console.log(`Loaded ${docs.length} documents.`);
  
  // Step 2: Split documents into chunks
  console.log('\n✂️ Splitting documents into chunks...');
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100
  });
  
  const splitDocs = await textSplitter.splitDocuments(docs);
  console.log(`Created ${splitDocs.length} chunks.`);
  
  // Step 3: Create in-memory vector store (no server needed)
  console.log('\n🧠 Creating in-memory vector store...');
  const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
  console.log('Vector store created successfully.');
  
  // Step 4: Create retriever
  const retriever = vectorStore.asRetriever({
    k: 3 // Number of documents to retrieve
  });
  
  // Step 5: Create prompt template
  const promptTemplate = PromptTemplate.fromTemplate(`
    Answer the question based only on the following context:

    {context}

    Question: {question}
    
    When answering, make sure to cite your source as [Document X] where X is the document number.
    If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
  `);
  
  // Step 6: Create RAG chain
  const ragChain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocumentsAsString),
      question: (input) => input.question
    },
    promptTemplate,
    llm,
    new StringOutputParser()
  ]);
  
  // Step 7: Execute queries
  await executeQuery(ragChain, 'What is RAG and what are its key components?');
  await executeQuery(ragChain, 'What are vector databases and how do they work?');
  await executeQuery(ragChain, 'How does embedding generation work in RAG systems?');
}

// Helper function to execute a query and display the result
async function executeQuery(chain, question) {
  console.log(`\n❓ Question: ${question}`);
  console.log('Retrieving relevant documents and generating answer...');
  
  try {
    const result = await chain.invoke({ question });
    console.log('\n🔍 Answer:');
    console.log(result);
  } catch (error) {
    console.error('Error generating response:', error);
  }
}

// Run the demo
runRAGDemo().catch(error => {
  console.error('Error in RAG demo:', error);
});
