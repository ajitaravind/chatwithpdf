import streamlit as st
from streamlit_chat import message

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain.schema import(
    HumanMessage,
    AIMessage
)

import os

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv(),override = True)

def load_document(file):
    import os
    name,extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx': #pip install docx2txt -q 
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        print(f'Loading {file}')
        loader = TextLoader(file)

    else:
        print('Document format is not supported')

    data = loader.load()
    return data

def chunk_data(data,chunk_size = 1024,chunk_overlap = 200):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(chunks,embeddings,persist_directory = "mydb")
    st.session_state.vector_store = vector_store
    return vector_store

def ask_and_get_answer(question,k = 4):
    from langchain_groq import ChatGroq

    from langchain.memory import ConversationBufferWindowMemory
    from langchain_community.chat_message_histories import SQLChatMessageHistory
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough,RunnableParallel
    from langchain.prompts import PromptTemplate
    from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    db3 = Chroma(persist_directory="mydb", embedding_function=embeddings)
    retriever = db3.as_retriever(search_type = 'similarity',search_kwargs={'k': k})

    rephraser_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Return only the question, nothing else.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
    
    rephraser_prompt = ChatPromptTemplate.from_template(rephraser_template)
    
    qa_template = """
        {rephrased_question}
        Below is relevant information you can use to answer the user question. Be as detailed as necessary \
        but for long answers, always try to give the response as bullet points. Ensure that you answer the user question\
        only from the given context. If you don't know the answer, say you don't know, don't invent answers.
        Context:
        {context}
        """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_template)
    
    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key = "chat_history", 
        chat_memory = SQLChatMessageHistory( 
        session_id="test_session", connection_string="sqlite:///sqlite.db"
        )
        ,
        return_messages = True,
        verbose = True
        )
    
    chain = RunnableParallel(
        {"question" : RunnablePassthrough(),
        "chat_history" : RunnablePassthrough()
        }
        ) |rephraser_prompt| llm | StrOutputParser()
    
    def get_context(question):
        result = chain.invoke({"question" : question,"chat_history" : memory.buffer})
        context = retriever.get_relevant_documents(result)
        return "\n".join([doc.page_content for doc in context])

    def get_result(question):
        result = chain.invoke({"question" : question,"chat_history" : memory.buffer})
        return result
    
    qa_chain = (
        RunnableParallel( 
        {
            "context": RunnableLambda(get_context),
            "rephrased_question" : RunnableLambda(get_result)
        } )
        | QA_CHAIN_PROMPT
        | llm
        |StrOutputParser()
            
        )
    
    result = qa_chain.invoke(question)
    memory.save_context({"input": question}, {"output": result})
    return result
    
    
def clear_history():
    # if 'content' in st.session_state:
    #     del st.session_state['content']
    st.session_state.messages = []
    st.session_state["input"] = ""


if __name__ == "__main__":
   
    import os
    from dotenv import load_dotenv,find_dotenv
    load_dotenv(find_dotenv(),override = True)

    st.set_page_config(
    page_title='Chat with any PDF',
    page_icon='🤖'
    )
    
   
    if 'uploaded_status' not in st.session_state:
        st.session_state.uploaded_status = ""
        
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = ""

    st.subheader('Chat with any PDF 🤖')
    st.write("")
    
    st.write(
    "Set environment variables:",
    os.environ["GROQ_API_KEY"] == st.secrets["GROQ_API_KEY"],
    )

    with st.sidebar:
        uploaded_file = st.file_uploader('Upload a file:', type = ['pdf','docx','txt'],accept_multiple_files = True)
        # print(uploaded_file)

        chunk_size = st.number_input('Chunk size:',min_value = 100,max_value = 2048,value = 1024,on_change = clear_history)
        k = st.number_input('k',min_value = 1, max_value = 20, value = 4,on_change = clear_history)
        add_data = st.button('Add Data',on_click = clear_history)
        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file...'):
                for i in range(len(uploaded_file)):
                    bytes_data = uploaded_file[i].read()
                    file_name = os.path.join('./',uploaded_file[i].name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)
                    data = load_document(file_name)
                    chunks = chunk_data(data,chunk_size=chunk_size)
                    vector_store = create_embeddings(chunks)
                    st.write(f'Embedding completed for file {file_name}')
                    st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully')
                st.session_state.uploaded_status = True
                print(st.session_state.uploaded_status)
                
       
    if st.session_state.uploaded_status == True:
        
        if "messages" not in st.session_state:
            st.session_state.messages = [AIMessage(content="Hello, how can I help you?")]  

        for message in st.session_state.messages:
            if isinstance(message, AIMessage):
                with st.chat_message("AI",avatar = "🤖"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human",avatar ="🙋‍♀️"):
                    st.write(message.content)   
        prompt = st.chat_input("How may I help you?")
        
        if prompt is not None and prompt != "":
            with st.chat_message("user",avatar = "🙋‍♀️"):
                st.markdown(prompt)
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.spinner('Getting the information for you!!!!'):
                response = ask_and_get_answer(prompt) 
                with st.chat_message("assistant", avatar = "🤖"):
                    st.markdown(response)
                st.session_state.messages.append(AIMessage(content=response))     
        
