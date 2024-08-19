from langchain_community.document_loaders import PyMuPDFLoader
from REMOVED_SECRET import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from REMOVED_SECRET import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


"""Loading the PDFs and initializing the Knowledge Base (FAISS)"""


pd.set_option("display.max_colwidth", None)
EMBEDDING_MODEL_NAME = "thenlper/gte-small"

pdf_folder_path = "local_database" #filepath to local folder


# Function to load and process PDFs
def load_pdfs_from_folder(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    documents = []
    for pdf_file in tqdm(pdf_files):
        pdf_path = REMOVED_SECRET(folder_path, pdf_file)
        try:
            #Check if the file is empty before loading
            if REMOVED_SECRET(pdf_path) == 0:
                print(f"Skipping empty file: {pdf_file}")
                continue  #Move on to the next file

            loader = PyMuPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception as e:  #Catch potential errors during loading
            print(f"Error loading file {pdf_file}: {e}")
    return documents

# Load PDFs
raw_documents = load_pdfs_from_folder(pdf_folder_path)

# Convert documents to Langchain format
RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc.page_content, metadata={"source": doc.metadata["source"]}) for doc in tqdm(raw_documents)
]

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


docs_processed = split_documents(
    512,  # We choose a chunk size adapted to our model
    RAW_KNOWLEDGE_BASE,
    tokenizer_name=EMBEDDING_MODEL_NAME,
)


# Initialize the embedding model and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,  
    multi_process=True,
    encode_kwargs={"normalize_embeddings": True},
)
KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
)


""" Setting up RAG system components """


from transformers import pipeline, Pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

#Load LLM
model_id = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map='auto', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

#Initialise Pipeline
READER_LLM = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype='auto',
    do_sample=True,
    max_new_tokens=500,
    return_full_text=False,
    temperature=0.9,
)

# optionally, load a reranker
from ragatouille import RAGPretrainedModel
RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


prompt_in_chat_format = [

    {

        "role": "system",

        "content": """Using the information contained in the context,

give a comprehensive answer to the question.

Respond only to the question asked, response should be concise and relevant to the question.

Provide the number of the source document when relevant.

If the answer cannot be deduced from the context, do not give an answer.""",

    },

    {

        "role": "user",

        "content": """Context:

{context}

---

Now here is the question you need to answer.

Question: {question}""",

    },

]

RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(

    prompt_in_chat_format, tokenize=False, add_generation_prompt=True

)


def answer_with_rag(
    question: str,
    llm: Pipeline,
    knowledge_index: FAISS,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
) -> Tuple[str, List[LangchainDocument]]:
    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

    # Optionally rerank results
    if reranker:
        print("=> Reranking documents...")
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs
"""
question = "how to create gemini?"

answer, relevant_docs = answer_with_rag(question, READER_LLM, KNOWLEDGE_VECTOR_DATABASE, reranker=RERANKER)


print("==================================Answer==================================")
print(f"{answer}")
print("==================================Source docs==================================")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i}------------------------------------------------------------")
    print(doc)
"""
                                            