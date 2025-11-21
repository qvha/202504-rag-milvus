#!/usr/bin/env python3

# on which card should i run
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys
import json
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_milvus import Milvus
from langchain.text_splitter import CharacterTextSplitter

prompt = PromptTemplate(
           input_variables=["context", "question"],
           template= "Answer the following question based on the provided context: "\
                     "{context}\n\nQuestion: {question}\nHelpful answer:"
         )

# load the dataset
with open("rag_instruct_qvha.json", 'r') as fhandle:
    dataset=json.load(fhandle)

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", 
    model_kwargs={"device": "cuda"},
    encode_kwargs={"batch_size": 32, "normalize_embeddings": True}
    )

llm_pipeline = pipeline("text-generation",
                        model="/models/Mistral-Small-3.1-24B-Instruct-2503",
                        max_new_tokens=200,
                        temperature=0.1)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

for i, MySlice in enumerate(dataset):
  #[
  #  {
  #    "question": "...",
  #    "answer": "...",
  #    "documents": [ "", "", "", ]
  question = MySlice["question"]

  chunks = list()
  for doc in  MySlice["documents"] : chunks+= text_splitter.split_text( doc )
  
  # inject the chunks in Milvus
  vectorstore = Milvus.from_texts(
    chunks,
    embedding_model,
    connection_args={"address": "10.197.177.115:19530" },  # location of the Milvus Vectorstore
    collection_name= f"rag_collection_{i}",                # each set of chunks has a unique ID
    drop_old=True                                          # delete if already there
  )
  
  retriever = vectorstore.as_retriever()                   # setup Milvus as the retriever
  qa_chain  = RetrievalQA.from_chain_type(                 # create the chain
    llm=llm,
    chain_type_kwargs={"prompt": prompt},
    retriever=retriever,
    verbose=True
  )

  # Finally, poping the question
  response = qa_chain.invoke(question)
  print(">>>> Inference <<<<<<\n", response["result"])     # print the full answer with context
  print(">>>> Reference <<<<<<\n", MySlice["answer"] )     # print the reference answer from the dataset
  
  if i==0: break
