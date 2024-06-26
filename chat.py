import streamlit as st
from streamlit_chat import message

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore
import pinecone

from langchain.schema import(
    HumanMessage,
    AIMessage
)

from langchain_community.chat_message_histories import SQLChatMessageHistory


import os
import time

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv(),override = True)

index_name = "genericdb"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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

def chunk_data(data,chunk_size = 2000,chunk_overlap = 200):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    print(f"I am length of chunks{len(chunks)}")
    return chunks

def create_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    import os
    from pinecone import Pinecone,ServerlessSpec

    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    if index_name not in pc.list_indexes().names():
        print(f'Creating the index {index_name} and the  embeddings')

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pc.create_index(
        name="genericdb",
        dimension=384, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
          cloud='aws', 
          region='us-east-1'
             ) 
        ) 
                     
    PineconeVectorStore.from_documents(
            chunks, embeddings, index_name=index_name
        )
    vector_store = PineconeVectorStore.from_existing_index(index_name, embeddings)
    return vector_store

     
def ask_and_get_answer(question,k = 4):
    from langchain_groq import ChatGroq

    from langchain.memory import ConversationBufferWindowMemory
    from langchain_community.chat_message_histories import SQLChatMessageHistory
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough,RunnableParallel
    from langchain.prompts import PromptTemplate
    from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    db3 =  st.session_state.vector_store
    retriever = db3.as_retriever(search_type = 'similarity',search_kwargs={'k': k})

    rephraser_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    If the question makes sense independently without additional context, respond with the original question.
    Examples: 
    "what are the key insights from the report" --> "what are the key insights from the report"
    "What is the impact of AI in healthcare" --> "What is the impact of AI in healthcare"
    
    Examples where the question needs to be rephrased.
    "What is its acceleration" --> "What is the acceleration of XYX"
    
    Important:
    Return only the question, nothing else.No need to give explaination or justification.
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
        chat_memory = st.session_state['chat_message_history']
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


def clear_chat_history(chat_message_history):
    if chat_message_history.messages:
        chat_message_history.clear()
        success = st.success("Chat history cleared successfully.")
        time.sleep(1) 
        success.empty()
    else:
        alert = st.warning("No chat history to clear.")
        time.sleep(1) 
        alert.empty() 

if __name__ == "__main__":
   
    import os
    from dotenv import load_dotenv,find_dotenv
    load_dotenv(find_dotenv(),override = True)

    st.set_page_config(
    page_title='Chat with any PDF',
    page_icon='🤖'
    )
    
        
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = ""

    if 'clear_history' not in st.session_state:
        st.session_state['clear_history'] = False  
        
    if 'chat_message_history' not in st.session_state: #sqlite history
        chat_message_history = SQLChatMessageHistory(
        session_id="test_session", connection_string="sqlite:///sqlite.db"
        )
        st.session_state['chat_message_history'] = chat_message_history     
    
    st.subheader('Chat with any PDF 🤖')
    st.write("")
    
    st.write(
    "Set environment variables:",
    os.environ["GROQ_API_KEY"] == st.secrets["GROQ_API_KEY"],
    os.environ["PINECONE_API_KEY"] == st.secrets["PINECONE_API_KEY"]
    )

    with st.sidebar:
        uploaded_file = st.file_uploader('Upload a file:', type = ['pdf','docx','txt'],accept_multiple_files = True)
        # print(uploaded_file)

        chunk_size = st.number_input('Chunk size:',min_value = 100,max_value = 2048,value = 2000,on_change = clear_history)
        k = st.number_input('k',min_value = 1, max_value = 20, value = 6,on_change = clear_history)
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
                    st.session_state.vector_store = vector_store
                    st.write(f'Embedding completed for file {file_name}')
                st.success('File uploaded, chunked and embedded successfully')
                print('File uploaded, chunked and embedded successfully')
                
    try:            
        vector_store = PineconeVectorStore.from_existing_index(index_name, embeddings)
        st.session_state.vector_store = vector_store
    except:
        vector_store = None
        st.session_state.vector_store = vector_store
    if  vector_store is not None:
        
        if st.button("Clear chat history"): 
            clear_chat_history(st.session_state['chat_message_history'])
        
        st.session_state['clear_history'] = True  
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
        
