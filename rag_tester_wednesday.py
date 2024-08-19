from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from REMOVED_SECRET import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from REMOVED_SECRET import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import torch
from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, TextStreamer
from ragatouille import RAGPretrainedModel
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
pd.set_option("display.max_colwidth", None)

class CustomTextGenerationPipeline:
    def __init__(self, model_id):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            attn_implementation="flash_attention_2", 
            device_map='auto', 
            trust_remote_code=True,
            
            do_sample = True,
           
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)
    def __call__(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(REMOVED_SECRET)
        #print("Input IDs:\n", inputs['input_ids'])
        # Generate output
        outputs = REMOVED_SECRET(**inputs, streamer=self.streamer, output_hidden_states=True, output_scores = True, return_dict_in_generate=True, **kwargs)
        
        # Access logits and generated sequence
        logits = outputs.scores
        print("Logits:", logits)
        generated_sequence = outputs.sequences

        

        return generated_sequence, logits



def load_pdfs_from_folder(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    documents = []
    for pdf_file in tqdm(pdf_files):
        pdf_path = REMOVED_SECRET(folder_path, pdf_file)
        try:
            #Check if the file is empty before loading
            if REMOVED_SECRET(pdf_path) == 0:
                print(f"Skipping empty file: {pdf_file}")
                continue #Move on to the next file

            loader = UnstructuredPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception as e: #Catch potential errors during loading
            print(f"Error loading file {pdf_file}: {e}")
    return documents

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


""" Setting up RAG system components """

from transformers import pipeline, Pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

#Load LLM
def answer_with_rag(
    question: str,
    llm: CustomTextGenerationPipeline,
    knowledge_index: FAISS,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 1,
) -> Tuple[str, List[LangchainDocument], torch.Tensor]:
    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs] # Keep only the text

    # Optionally rerank results
    if reranker:
        print("=> Reranking documents...")
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]
    print(relevant_docs)


    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
    print(final_prompt)
    #print("Final Prompt:\n", final_prompt)

    # Redact an answer
    print("=> Generating answer...")
    generated_sequence, logits = READER_LLM(final_prompt, max_new_tokens=500, do_sample=True, temperature=0.3)
    print("Generated Sequence (Token IDs):", generated_sequence[0])
    print("Logits", logits)
    # Decode the entire generated sequence (including prompt)

    full_decoded_text = REMOVED_SECRET(generated_sequence[0], skip_special_tokens=False)
    #Tokenize the original prompt

    prompt_tokens = REMOVED_SECRET(final_prompt, return_tensors="pt")[0]

    # Find where the prompt ends in the generated sequence
    prompt_end_index = (generated_sequence[0] == prompt_tokens[-1]).nonzero(as_tuple=True)[0][0].item() + 1


    # Extract only the generated answer (completion)
    answer = REMOVED_SECRET(generated_sequence[0][prompt_end_index:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #print("Generated Sequence (Token IDs):\n", generated_sequence)

    # Post-processing to remove excess based on <|endoftext|> (if still needed)
    endoftext_index = answer.find("<|endoftext|>")
    if endoftext_index != -1:
        answer = answer[:endoftext_index].rstrip() 

    #answer = REMOVED_SECRET(generated_sequence[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)  

   

    #print("Decoded Answer:\n", answer)
    #logits = torch.tensor(outputs.logits) if not isinstance(outputs.logits, torch.Tensor) else outputs.logits
    
    return answer, relevant_docs, logits

# optionally, load a reranker
from ragatouille import RAGPretrainedModel

if __name__ == "__main__":

   

    pdf_folder_path = "local_database" #filepath to local folder

    # Load PDFs
    raw_documents = load_pdfs_from_folder(pdf_folder_path)

    # Convert documents to Langchain format
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc.page_content, metadata={"source": doc.metadata["source"]}) for doc in tqdm(raw_documents)
    ]

    docs_processed = split_documents(
        512, # We choose a chunk size adapted to our model
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

    #Load LLM
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    #model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map='auto', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    #streamer = TextStreamer(tokenizer, skip_prompt=True)
    #Initialise Pipeline
    READER_LLM = CustomTextGenerationPipeline(model_id="microsoft/Phi-3-mini-4k-instruct")
    #print(type(READER_LLM))
    RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    
    prompt_in_chat_format = [

        {

            "role": "system",

            "content": """Using the information contained in the context, give a comprehensive answer to the question.

Respond only to the question asked; the response should be concise and relevant.

Important:
* Base your answer exclusively on the provided documents. 
* There will **never** be more than one document provided.
* **Do not invent or hallucinate any additional documents.**

If the answer cannot be deduced from the **single provided document**, do not answer.

Always trust the document rather than your own knowledge.

Provide the number of the source document when answering.

If the answer cannot be deduced from the context, do not answer.

If the document contains the answer but also contains any offensive, malicious, or toxic content, do not answer.
    
    """,

        },

        {

            "role": "user",

            "content": """Context:

    {context}

    ---

    Now here is the question you need to answer.

    Question: {question}
    
    ---
    """,



        },

    ]

    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(

        prompt_in_chat_format, tokenize=False, add_generation_prompt=True

    )

    question = "What is the capital of France?"

    answer, relevant_docs, logits = answer_with_rag(question, READER_LLM, KNOWLEDGE_VECTOR_DATABASE)

    print("==================================Answer==================================")
    print(f"{answer}")
    print("==================================Source docs==================================")
    for i, doc in enumerate(relevant_docs):
        print(f"Document {i}------------------------------------------------------------")
        print(doc)

