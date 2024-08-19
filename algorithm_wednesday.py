import os
import random
import re
import fitz
import tiktoken
import torch
import REMOVED_SECRET as F
import numpy as np
from keybert import KeyBERT
from tkinter import filedialog
import tkinter as tk
from langchain_community.vectorstores import FAISS
from REMOVED_SECRET import DistanceStrategy
from langchain.document_loaders import PyMuPDFLoader
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

from rag_for_notebook import READER_LLM, KNOWLEDGE_VECTOR_DATABASE, RERANKER, CustomTextGenerationPipeline, answer_with_rag, docs_processed

# Initialize models
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model_id = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map='cuda', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
encoding = tiktoken.encoding_for_model("text-embedding-3-small") #openai embedding model (used for vocab)

vocab_size = 50257 #random but large guess at vocab size
vocab_list = []

# iterate through a range of possible token indices
for token_id in range(vocab_size):
    try:
        #decode the token_id to get the corresponding token string
        token = encoding.decode([token_id])
        # add the token to the vocabulary list
        vocab_list.append(token)
    except KeyError:
        #if decoding fails, it's likely an out-of-vocabulary token or a special token
        
        pass

print(f"Estimated Vocabulary size: {len(vocab_list)}")
#^ get vocabulary from tiktoken (openai)


def query_rag_system(question, llm=READER_LLM, knowledge_index = KNOWLEDGE_VECTOR_DATABASE, reranker=RERANKER):
    """
    Queries the RAG system with the given question.

    Args:
        question: The question to ask the RAG system.
        llm: The language model to use for answer generation (default: READER_LLM).
        knowledge_index: The vector store containing the document embeddings (default: KNOWLEDGE_VECTOR_DATABASE).
        reranker: The reranker model (optional, default: RERANKER).

    Returns:
        A tuple containing:
            - answer: The generated answer from the RAG system.
            - relevant_docs: The list of retrieved relevant documents.
            - logits: The logits of the generated answer.
    """

    # query the RAG system and get the answer, relevant docs, and logits
    answer, relevant_docs, logits = answer_with_rag(
        question=question,
        llm=llm,
        knowledge_index=knowledge_index,
        reranker=reranker
    )

    return answer, relevant_docs, logits



def find_strongest_keyword(keywords, chunk_text):
    """
    Finds the keyword with the highest average semantic similarity to the chunk's context.
    """

    # encode the chunk and keywords using the embedding model
    chunk_embedding = embedding_model.encode(chunk_text, convert_to_tensor=True)
    keyword_embeddings = [embedding_model.encode(kw, convert_to_tensor=True) for kw in keywords]

    # calculate cosine similarity between each keyword and the chunk
    keyword_similarities = {kw: 0 for kw in keywords}  # Initialize similarities
    for kw, kw_embedding in zip(keywords, keyword_embeddings):
        if kw in chunk_text:  # Check if the keyword is present in the chunk
            similarity = util.pytorch_cos_sim(chunk_embedding, kw_embedding).item()
            keyword_similarities[kw] = similarity

    # Find the keyword with the highest similarity
    if keyword_similarities:
        strongest_keyword = max(keyword_similarities, key=keyword_similarities.get)
        return strongest_keyword
    else:
        return None   # No keywords found in the chunk


def inject_text_into_pdf(input_pdf_path, output_pdf_path, text_to_inject, keywords_list, docs_processed):
    """
    Injects adversarial text into the selected PDF.

    Args:
    input_pdf_path: selected PDF path from browse_for_pdf function
    output_pdf_path: desired location of new PDF
    text_to_inject: the adversarial sequence to be injected
    keywords_list: a list of keywords from the selected PDF to push into find_strongest_keyword
    docs_processed: embedded processed documents (one pdf can be multiple docs)
    
    """
    pdf_document = fitz.open(input_pdf_path)

    # Create the zero-width version of the injected word
    zero_width_inject_word = "\u200B".join(list(text_to_inject))

    for doc in docs_processed:
       # page_num = doc.metadata['page'] 
        page_num = 0
        page = pdf_document[page_num] 
        original_text = page.get_text("text")

        # Find keywords within this chunk
        chunk_keywords = [kw for kw in keywords_list if kw in doc.page_content]

        if chunk_keywords:
            # Find the keyword with the highest semantic strength (you'll need to implement this)
            strongest_keyword = find_strongest_keyword(chunk_keywords, doc.page_content) 

            # Inject before the strongest keyword
            new_text = original_text.replace(strongest_keyword, f"{zero_width_inject_word}{strongest_keyword}")

            page.clean_contents()
            page.insert_text((0, 0), new_text, fontsize=12)

        print(f"Processing page {page_num + 1} of {len(pdf_document)}")

    pdf_document.save(output_pdf_path)
    pdf_document.close()
    print("Injection complete!")


def extract_keywords_from_pdf(pdf_path, num_keywords=50):

    keywords_list = []
    try:
        loader = PyMuPDFLoader(pdf_path)
        document = loader.load()[0]
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(document.page_content, keyphrase_ngram_range=(1, 3), top_n=num_keywords)
        keywords_list = [keyword for keyword, score in keywords]
    except Exception as e:
        print(f"Error loading or processing PDF {pdf_path}: {e}")
    return keywords_list

def browse_for_pdf():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    return file_path

def weighted_loss(logits, t_res, crucial_indices, weight=0.5):
    """
    Calculates the weighted loss.
    """
    loss = F.cross_entropy(logits, t_res)
    crucial_logits = logits[:, crucial_indices]
    crucial_t_res = t_res[crucial_indices]
    crucial_loss = F.cross_entropy(crucial_logits, crucial_t_res)
    weighted_loss = loss * (1 - weight) + crucial_loss * weight
    return weighted_loss

def mutate_seq_with_gradient(seq_tokens, logits, target_response_tokens, crucial_indices, weight=0.8, k=32, learning_rate=0.1):
    """
    Mutates the sequence based on the gradient of the weighted loss.

    Args:
        seq_tokens: The current attack sequence (tokenized).
        logits: The raw output of the LLM before the final softmax layer.
        target_response_tokens: The targeted malicious response (tokenized).
        crucial_indices: Indices of the crucial tokens in the target response.
        weight: The weight assigned to the crucial loss component.
        k: The number of new sequences to generate.
        learning_rate: The learning rate for gradient descent.

    Returns:
        A list of k mutated sequences.
    """

    # Calculate the weighted loss and its gradient
    loss = weighted_loss(logits, target_response_tokens, crucial_indices, weight)
    loss.backward() 

    # Get the gradient with respect to the embedded input sequence
    seq_embeddings = REMOVED_SECRET()(seq_tokens) 
    grad = REMOVED_SECRET.clone() 

    new_seqs = []
    for _ in range(k):
        # Randomly select a token to mutate
        mutate_index = torch.randint(0, len(seq_tokens[0]), (1,)).item()

        # Mutate the token's embedding based on the gradient
        mutated_embedding = seq_embeddings[0][mutate_index] - learning_rate * grad[0][mutate_index]

        # Find the closest token in the vocabulary to the mutated embedding
        distances = torch.norm(REMOVED_SECRET().weight.data - mutated_embedding, dim=1)
        closest_token_id = torch.argmin(distances).item()

        # Create a new sequence with the mutated token
        new_seq = seq_tokens.clone()
        new_seq[0][mutate_index] = closest_token_id
        new_seqs.append(new_seq)

    # Zero out the gradients for the next iteration
    REMOVED_SECRET()

    return new_seqs


def choose_best_sequence(new_seqs, losses):
    """
    Chooses the best sequence from the mutated candidates based on their pre-calculated losses.

    Args:
        new_seqs: A list of mutated sequences (tokenized).
        losses: A list of corresponding loss values for each sequence.

    Returns:
        The best sequence (tokenized) with the lowest loss.
    """
    
    best_index = np.argmin(losses)
    return new_seqs[best_index]

# Main Workflow
pdf_file_path = browse_for_pdf()
if pdf_file_path:
    keyword_results = extract_keywords_from_pdf(pdf_file_path)
    if keyword_results:
        best_keywords = keyword_results[:5]
        initial_sub_document = best_keywords
        initial_vocab = vocab_list
        token_vocabulary = initial_vocab + keyword_results
        target_response = "I'm sorry, I can't help with that."
        query_based_on_pdf = "What is the capital of France?"

        T = 5  
        B = 3  

        target_response_tokens = tokenizer.encode(target_response, return_tensors='pt').squeeze().to(model.device)
        crucial_phrase = "I cannot provide an answer"
        crucial_indices = [i for i, token in enumerate(target_response_tokens) if tokenizer.decode([token]) in crucial_phrase.split()]

        sub_document = initial_sub_document.copy()
        rag_database_folder = "local_database"
        # Encode documents into embeddings
       
        embeddings = embedding_model.encode([doc.page_content for doc in docs_processed])

            # Combine texts and embeddings into tuples
        text_embeddings = list(zip([doc.page_content for doc in docs_processed], embeddings))

            # Create FAISS index from embeddings
        knowledge_index = FAISS.from_embeddings(
                text_embeddings, embedding_model, metadatas=[doc.metadata for doc in docs_processed], distance_strategy=DistanceStrategy.COSINE
            )

        for i in range(T):
            l = random.randint(0, len(sub_document) - 1)
            candidate_sub_documents = []
            losses = []

            for b in range(B):
                new_token = random.choice(token_vocabulary)
                candidate = sub_document[:l] + [new_token] + sub_document[l+1:]
                
                # Inject candidate into the PDF at the strongest keyword locations
                output_pdf_path = REMOVED_SECRET(rag_database_folder, f"updated_pdf_{i}_{b}.pdf")
                inject_text_into_pdf(pdf_file_path, output_pdf_path, ' '.join(candidate), keyword_results, docs_processed)
                #re-load modified pdf and upate keywords & vocab
                pdf_file_path = output_pdf_path
                keyword_results = extract_keywords_from_pdf(pdf_file_path)
                token_vocabulary = initial_vocab + keyword_results

                embeddings = embedding_model.encode([doc.page_content for doc in docs_processed])
                text_embeddings = list(zip([doc.page_content for doc in docs_processed], embeddings))
                knowledge_index = FAISS.from_embeddings(  # Recreate the index
                        text_embeddings, embedding_model, metadatas=[doc.metadata for doc in docs_processed], distance_strategy=DistanceStrategy.COSINE
                    )



                #query RAG system and get logits
                print(f"Type of query_based_on_pdf: {type(query_based_on_pdf)}")
                print(f"Type of updated_knowledge_index: {type(knowledge_index)}")
                answer, _, logits = query_rag_system(query_based_on_pdf, knowledge_index=knowledge_index)
                print(f"Iteration {i+1}/{T}, Batch {b+1}/{B}: Candidate response: {answer}")

                # Calculate weighted loss
                loss = weighted_loss(logits, target_response_tokens, crucial_indices)
                print(f"Loss: {loss:.4f}")
                

                # Mutate the sequence based on the gradient and loss
                candidate_tokens = tokenizer.encode(' '.join(candidate))
                candidate_tokens = torch.tensor([candidate_tokens]).to(model.device)
                new_seqs = mutate_seq_with_gradient(candidate_tokens, logits, target_response_tokens, crucial_indices)
                 # Store the loss value
                losses.append(loss.item())
                # Evaluate the new sequences and select the best one (replace ... with your actual evaluation logic)
                best_seq = choose_best_sequence(new_seqs) # You'll need to implement this)

                # Update candidate based on the best sequence
                candidate = tokenizer.decode(best_seq[0], skip_special_tokens=True).split()
                candidate_sub_documents.append(candidate)
               

            # Select the best candidate based on loss (lower loss is better)
            best_candidate_index = np.argmin(losses)
            sub_document = candidate_sub_documents[best_candidate_index]
            print(f"Iteration {i+1}/{T}: Best candidate sub-document: {' '.join(sub_document)} (Loss: {losses[best_candidate_index]:.4f})")
            # Early stopping if loss is below a threshold (optional)
            if losses[best_candidate_index] < 0.4:  # Adjust the threshold as needed
                    break

        final_sub_document_text = ' '.join(sub_document)
        print(f"Final optimized sub-document: {final_sub_document_text}")

        final_response_file = "final_response.txt"
        with open(final_response_file, "w") as f:
            f.write(final_sub_document_text)
        print(f"Final response saved to {final_response_file}")

    else:
        print("No keywords extracted from the PDF.")
else:
    print("No file selected.")
