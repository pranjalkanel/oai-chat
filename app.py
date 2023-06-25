import streamlit as st
import pickle
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title("PDF Summary")
    st.markdown('''
    ## About
    A web app based on LLM to summarize a PDF
    ''')
    add_vertical_space(5)
    st.write("@[bhattsapp](https://linkedin.com/in/kishor-bhatt)")


def main():
    os.environ["OPENAI_API_KEY"] = os.getenv("oai_skey")
    os.environ["OPENAI_ORGANIZATION"] = os.getenv("oai_okey")

    st.header("Talk with PDF ...")
    pdf = st.file_uploader("Upload the PDF file...",type="pdf")

    if pdf is not None: 
        pdf_reader = PdfReader(pdf)
        text= ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)
        st.write(chunks)
        
        # create embedding objects 
        pdf_name = pdf.name[:-4]
        st.write(pdf_name)
        
        if os.path.exists(f"{pdf_name}.pkl"):
            with open(f"{pdf_name}.pkl","rb") as f:
                VectorStore = pickle.load(f) 
        else:
            print("Making API request ::: ")
            with get_openai_callback() as cb:
                print("Inside callback")
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
                print(cb)
            with open(f"{pdf_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)

        # prompts
        query = st.text_input("Ask question on PDF ...")
        # st.write(query)
        if query:
            docs = VectorStore.similarity_search(query=query,k=3)
            # st.write(docs)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,question=query)
                print(cb)
            st.write(response)

if __name__ =='__main__':
    main()