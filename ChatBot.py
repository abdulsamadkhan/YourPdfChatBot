import streamlit as st
from dotenv import load_dotenv
import databutton as db
import pickle
import pdfplumber


from PyPDF2 import PdfReader

from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import faiss
import os
import openai
import io



# Sidebar contents
with st.sidebar:
    st.title("🤗💬Pdf Chat bot💬🤗")
    st.write("Enter the OPENAI key, this will not be saved")
    api = st.sidebar.text_input("API_Key", type="password")
    #os.environ["OPENAI_API_KEY"] = api
    openai.api_key = api
    # st.write(api)
    st.markdown(
        """
    ## About
    - This is a chatbot which answers the questions about the pdf you have loaded 
    
    - if the same pdf is loaded again, the stored database (VectorStore) will be loaded:
 
    """
    )
    add_vertical_space(1)
    st.write("Made by Abdul Samad, https://github.com/abdulsamadkhan/YourPdfChatBot")


#load_dotenv()


def main():
    st.header("Chat with PDF 💬")
    # checks if the API Key is Entered
    if api:
        # upload a PDF file
        pdf = st.file_uploader("Upload your PDF", type="pdf")

        if pdf is not None:
            pdf_reader = PdfReader(pdf)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            # st.write(text)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100, length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # embeddings
            store_name = pdf.name[:-4]
            store_name = store_name.replace(
                " ", ""
            )  # Replace space and "#" characters with underscores or remove them

            # st.write(f"{store_name}")
            # st.write(chunks)

            try:
                if db.storage.binary.get(f"{store_name}.pkl"):
                    # raise FileNotFoundError(f"{store_name}.pkl not found")
                    findex = db.storage.binary.get(f"{store_name}.index")
                    st.write("Your Vector database for this file already exists !!")
                    with open("docs.index", "wb") as file:
                        file.write(findex)
                    index = faiss.read_index("docs.index")
                    VectorStore = pickle.loads(
                        db.storage.binary.get(f"{store_name}.pkl")
                    )
                    VectorStore.index = index

                else:
                    pass

            except FileNotFoundError as e:
                # Custom exception handling
                st.write(f"This pdf embedding does not exist in the database: {str(e)}")
                # Additional actions if needed
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

                #st.write("Vector Store create")

                # Write index to file (no method to generate stream object instead)
                faiss.write_index(VectorStore.index, "docs.index")
                # st.write(f"{store_name}.index")
                # Open the file and dump to Databutton storage
                with open("docs.index", "rb") as file:
                    db.storage.binary.put(f"{store_name}.index", file.read())
                # This little hack I got from HW Chase's repo on Notion Q&A
                VectorStore.index = None
                # Store the pickled object too
                db.storage.binary.put(f"{store_name}.pkl", pickle.dumps(VectorStore))

                # Accept user questions/query
            query = st.text_input("Ask questions about your PDF file:")
            # st.write(query)

            if query:
                docs = VectorStore.similarity_search(query=query, k=3)
                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    st.write(cb)
                st.write(response)


if __name__ == "__main__":
    main()
