import os
import logging
import traceback
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import joblib
from langchain.chat_models import init_chat_model

# Configure Logger
logging.basicConfig(
    filename="rag_initializer.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class RAGInitializer:
    def __init__(self):
        """
        Initialize Conversational Retrieval-Augmented Generation for Ubuntu Image Building.
        """

        try:
            logging.info("Initializing RAG system...")

            # Load embeddings and vector store
            self.embeddings = self._load_embeddings()
            self.vectorstore = self._load_vectorstore()

            # Set API key and initialize LLM
            os.environ["GROQ_API_KEY"] = "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"
            self.llm = self._init_llm()

            # Create conversation chain
            self.conversation_chain = self._create_conversation_chain()

            # Chat history
            self.chat_history = []

            logging.info("RAG system initialized successfully.")

        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            logging.debug(traceback.format_exc())
            raise RuntimeError("Failed to initialize RAG system. Check logs for details.")

    def _load_embeddings(self):
        """Load embeddings from a serialized file."""
        try:
            logging.info("Loading embeddings...")
            embeddings = joblib.load("resources/embedding.pkl")
            logging.info("Embeddings loaded successfully.")
            return embeddings
        except FileNotFoundError:
            logging.error("Embeddings file not found.")
            raise FileNotFoundError("Embeddings file missing: resources/embedding.pkl")
        except Exception as e:
            logging.error(f"Error loading embeddings: {e}")
            logging.debug(traceback.format_exc())
            raise

    def _load_vectorstore(self):
        """Load vector store from a serialized file."""
        try:
            logging.info("Loading vector store...")
            vectorstore = joblib.load("resources/vectorstore.pkl")
            logging.info("Vector store loaded successfully.")
            return vectorstore
        except FileNotFoundError:
            logging.error("Vector store file not found.")
            raise FileNotFoundError("Vector store file missing: resources/vectorstore.pkl")
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
            logging.debug(traceback.format_exc())
            raise

    def _init_llm(self):
        """Initialize the LLM with Groq API."""
        try:
            logging.info("Initializing LLM...")
            model = init_chat_model("llama3-8b-8192", model_provider="groq")
            logging.info("LLM initialized successfully.")
            return model
        except Exception as e:
            logging.error(f"Error initializing LLM: {e}")
            logging.debug(traceback.format_exc())
            raise

    def _create_conversation_chain(self) -> ConversationalRetrievalChain:
        """
        Create a conversational retrieval chain with custom prompt.
        """
        try:
            logging.info("Creating conversational retrieval chain...")

            # Custom prompt to focus on Ubuntu image building
            qa_prompt = PromptTemplate(
                template="""You are an expert assistant helping with Ubuntu image building and related tasks.
                
                Context Guidelines:
                - Use ONLY the provided context to answer the question
                - If the context does not contain enough information, clearly state that
                - Provide precise, technical answers
                - Reference specific parts of the context when possible
                
                
                Current Context:
                {context}
                
                Question: {question}
                
                Helpful, Precise Technical Answer:""",
                                input_variables=["context", "question"]
            )

            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}, return_source_documents=True),
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": qa_prompt}
            )

            logging.info("Conversational retrieval chain created successfully.")
            return chain

        except Exception as e:
            logging.error(f"Error creating conversation chain: {e}")
            logging.debug(traceback.format_exc())
            raise RuntimeError("Failed to create conversation chain. Check logs for details.")

    def process_query(self, question: str) -> Dict[str, Any]:
        """
        Process a query with streaming and source retrieval.

        :param question: User's input question
        :return: Dictionary with response structure
        """
        try:
            logging.info(f"Processing query: {question}")

            # Run conversation chain with streaming
            result = self.conversation_chain.invoke(
                {
                    "question": question,
                    "chat_history": self.chat_history
                }
            )

            # Prepare response in the specified structure
            response = {
                "question": question,
                "chat_history": self.chat_history,
                "answer": result["answer"],
                "source_documents": result.get("source_documents", [])
            }

            # Update chat history
            self.chat_history.append((question, response["answer"]))
            logging.info("Query processed successfully.")
            return response

        except KeyError as ke:
            logging.error(f"Missing key in response: {ke}")
            logging.debug(traceback.format_exc())
            return {
                "question": question,
                "chat_history": self.chat_history,
                "answer": "Sorry, I couldn't process your request due to a response format issue.",
                "source_documents": []
            }

        except Exception as e:
            logging.error(f"Error processing query: {e}")
            logging.debug(traceback.format_exc())
            return {
                "question": question,
                "chat_history": self.chat_history,
                "answer": "Sorry, I couldn't process your request.",
                "source_documents": []
            }

