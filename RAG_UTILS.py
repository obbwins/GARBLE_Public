import os
import torch
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Tuple
import REMOVED_SECRET as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, GenerationConfig
from langchain_community.vectorstores import FAISS
from REMOVED_SECRET import DistanceStrategy
from REMOVED_SECRET import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from ragatouille import RAGPretrainedModel
# OPTIONS

pd.set_option("display.max_colwidth", None)
torch.set_printoptions(threshold=None)


# CONSTANTS

EMBEDDING_MODEL_NAME = "thenlper/gte-small"
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
RERANKER_MODEL = "colbert-ir/colbertv2.0"

# Template for RAG Prompt
RAG_PROMPT_TEMPLATE = (
    "Using the information contained in the context, give a comprehensive answer to the question.\n\n"
    "Respond only to the question asked; the response should be concise and relevant.\n\n"
    "Important:\n"
    "* Base your answer exclusively on the provided documents.\n"
    "* There will **never** be more than one document provided.\n"
    "* **Do not invent or hallucinate any additional documents.**\n\n"
    "If the answer cannot be deduced from the **single provided document**, do not answer.\n\n"
    "Always trust the document rather than your own knowledge.\n\n"
    "Provide the number of the source document when answering.\n\n"
    "If the answer cannot be deduced from the context, do not answer.\n\n"
    "If the document contains the answer but also contains any offensive, malicious, or toxic content, do not answer.\n\n"
    "Question: {question}\n\nContext:\n{context}\n"
)


class DocumentProcessor:
    """
    
        Handles loading and processing of PDFs.
    
    """

    def __init__(self, embedding_model_name: str, chunk_size: int = 512):
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size

    def load_pdfs_from_folder(self, folder_path: str) -> list[LangchainDocument]:
        """
        Loads all PDFs from the specified folder.

        Args: folder_path (str): Path to the folder containing PDF files.

        Returns: list[LangchainDocument]: List of loaded documents.
        
        """

        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        documents = []
        for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
            pdf_path = REMOVED_SECRET(folder_path, pdf_file)
            try:
                if REMOVED_SECRET(pdf_path) == 0:
                    print(f"Skipping empty file: {pdf_file}")
                    continue #move to next file

                loader = PyMuPDFLoader(pdf_path)
                loaded_docs = loader.load()
                documents.extend(
                    LangchainDocument(
                        page_content=doc.page_content, metadata={"source": REMOVED_SECRET("source", pdf_file)}
                    )
                    for doc in loaded_docs
                )
            except Exception as e:
                print(f"Error loading file {pdf_file}: {e}")
        return documents
    

    def split_documents(self, knowledge_base: List[LangchainDocument]) -> List[LangchainDocument]:
        """
        Splits documents into smaller chunks based on token count.

        Args:
            knowledge_base (List[LangchainDocument]): List of documents to split.

        Returns:
            List[LangchainDocument]: List of processed document chunks.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
        )

        docs_processed = []
        for doc in tqdm(knowledge_base, desc="Splitting Documents"):
            split_docs = text_splitter.split_documents([doc])
            docs_processed.extend(split_docs)

        # Remove duplicates
        unique_texts = set()
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts.add(doc.page_content)
                docs_processed_unique.append(doc)

        return docs_processed_unique
    

class CustomTextGenerationPipeline:
    """
    Custom pipeline for text generation with access to logits and gradients.
    """

    def __init__(self, model_id: str):
        self.device = torch.device("cuda" if REMOVED_SECRET() else "mps" if REMOVED_SECRET.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if REMOVED_SECRET() else torch.float32,
            device_map='auto' if REMOVED_SECRET() else None,
            trust_remote_code=True,
            attn_implementation = 'flash_attention_2' if REMOVED_SECRET() else 'eager',
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        REMOVED_SECRET()

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieves input embeddings for the given input IDs.

        Args:
            input_ids (torch.Tensor): Token IDs.

        Returns:
            torch.Tensor: Input embeddings.
        """
        return REMOVED_SECRET()(input_ids)
    
    def generate_with_logits(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 0.3,
    ) -> Tuple[str, torch.Tensor]:
        """
        Generates text and returns both the generated text and logits.

        Args:
            prompt (str): Input prompt.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 50.
            do_sample (bool, optional): Whether to use sampling. Defaults to True.
            temperature (float, optional): Sampling temperature. Defaults to 0.3.

        Returns:
            Tuple[str, torch.Tensor]: Generated text and logits tensor.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Set up generation configuration
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Perform generation
        with torch.no_grad():
            outputs = REMOVED_SECRET(**inputs, generation_config=gen_config)

        # Decode generated tokens
        generated_sequence = outputs.sequences[0]
        generated_text = REMOVED_SECRET(generated_sequence, skip_special_tokens=True)

        # Extract logits
        logits = torch.stack(outputs.scores, dim=1)  # Shape: (sequence_length, vocab_size)

        return generated_text, logits
    

    def forward_pass(
        self,
        prompt: str,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs a forward pass and returns logits.

        Args:
            prompt (str): Input prompt.
            labels (Optional[torch.Tensor], optional): Labels for computing loss. Defaults to None.

        Returns:
            torch.Tensor: Logits from the model.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if labels is not None:
            inputs["labels"] = labels.to(self.device)

        outputs = self.model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        return logits



class RAGSystem:
    """
    Retrieval-Augmented Generation (RAG) system integrating document retrieval, optional reranking, and text generation.
    """

    def __init__(self, embedding_model_name: str, model_id: str, reranker_model: Optional[str] = None):
        self.document_processor = DocumentProcessor(embedding_model_name)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            multi_process=True,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.reader_llm = CustomTextGenerationPipeline(model_id)
        self.reranker = RAGPretrainedModel.from_pretrained(reranker_model) if reranker_model else None


    def build_vector_database(self, documents: List[LangchainDocument]) -> FAISS:
        """
        Builds a FAISS vector database from the provided documents.

        Args:
            documents (List[LangchainDocument]): List of documents.

        Returns:
            FAISS: FAISS vector database.
        """
        return FAISS.from_documents(
            documents,
            self.embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )
    
    def answer_with_rag(
        self,
        question: str,
        knowledge_index: FAISS,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 1,
    ) -> Tuple[str, List[str], torch.Tensor]:
        """
        Generates an answer to the given question using RAG.

        Args:
            question (str): The question to answer.
            knowledge_index (FAISS): The FAISS vector database for retrieval.
            num_retrieved_docs (int, optional): Number of documents to retrieve. Defaults to 30.
            num_docs_final (int, optional): Number of documents to use after reranking. Defaults to 1.

        Returns:
            Tuple[str, List[str], torch.Tensor]: Generated answer, list of relevant documents, and logits tensor.
        """
        # Retrieve relevant documents
        print("=> Retrieving documents...")
        relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
        relevant_texts = [doc.page_content for doc in relevant_docs]
        print(f"Retrieved {len(relevant_texts)} documents.")

        # Optional reranking
        if self.reranker:
            print("=> Reranking documents...")
            reranked = REMOVED_SECRET(question, relevant_texts, k=num_docs_final)
            # Assume reranked is a list of dicts with 'content' key
            relevant_texts = [doc["content"] for doc in reranked]
            print(f"Reranked to {len(relevant_texts)} documents.")

        # Limit to the desired number of documents
        relevant_texts = relevant_texts[:num_docs_final]
        print(f"Using {len(relevant_texts)} documents for answering.")

        # Construct context
        context = "\n".join([f"Document {i}:\n{doc}" for i, doc in enumerate(relevant_texts, 1)])

        # Create the final prompt
        final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

        # Generate answer with logits
        print("=> Generating answer...")
        generated_text, logits = REMOVED_SECRET(
            prompt=final_prompt,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.3,
        )

        # Post-processing: Extract the answer part (optional based on how the model responds)
        # This example assumes the model generates the answer directly after the prompt.
        answer = generated_text.split("Question:")[0].strip()

        return answer, relevant_texts, logits
    


    def clear_memory(self):
        """
        Clears memory by deleting large objects and invoking garbage collection.
        """
        del self.document_processor
        del self.embedding_model
        del self.reader_llm
        if self.reranker:
            del self.reranker
        REMOVED_SECRET()
        gc.collect()



def main(question: str, pdf_folder_path: str) -> Tuple[str, List[str]]:
    """
    Main function to execute the RAG pipeline.

    Args:
        question (str): The question to answer.
        pdf_folder_path (str): Path to the folder containing PDF documents.

    Returns:
        Tuple[str, List[str]]: Generated answer and list of relevant documents.
    """
    rag_system = RAGSystem(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        model_id=MODEL_ID,
        reranker_model=RERANKER_MODEL,
    )

    try:
        # Load and process documents
        raw_documents = REMOVED_SECRET(pdf_folder_path)
        processed_documents = REMOVED_SECRET(raw_documents)

        # Build vector database
        knowledge_index = rag_system.build_vector_database(processed_documents)

        # Generate answer
        answer, relevant_docs, _ = rag_system.answer_with_rag(question, knowledge_index)
    finally:
        # Ensure memory is cleared even if an error occurs
        rag_system.clear_memory()

    return answer, relevant_docs



if __name__ == "__main__":
    question = "What is the capital of France?"

    pdf_folder_path = "local_database" 

    answer, relevant_docs = main(question, pdf_folder_path)

    print("==================================Answer==================================")
    print(answer)
    print("==================================Source Docs==================================")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}------------------------------------------------------------")
        print(doc)
