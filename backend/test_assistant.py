import urllib3
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # Messaging utilities from langchain_core
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Templating utilities for chat from langchain_core
from langchain_openai import ChatOpenAI  # OpenAI interface from langchain_openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import datetime
import os
CONDENSE_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

<chat_history>
  {chat_history}
</chat_history>

Follow Up Input: {question}
Standalone question:"""
# If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
# If the question is not related to the context or chat history, politely respond that you are tuned to only answer questions that are related to the context.
QA_TEMPLATE = """You are an expert report analyst. Use the following pieces of context to answer the question at the end.
Current time: {current_time}.
Pleaes answer to the question considering the context.

<context>
  {context}
</context>

<chat_history>
  {chat_history}
</chat_history>

Question: {question}
Answer:
"""
class EducatorAssistant:
    def __init__(self, pinecone_index):
    
        self.model = ChatOpenAI(temperature=0, model="gpt-4o")
        self.index_name  =  os.getenv("PINECONE_INDEX_NAME")
        self.embed_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.history = []
# Define a system message template
 
        self.pinecone_index = pinecone_index
        # Build a prompt template for the conversation with placeholders and system messages
        self.prompt_qa = ChatPromptTemplate.from_messages([
            ("system", QA_TEMPLATE),
            
        ])
        self.prompt_condense = ChatPromptTemplate.from_messages([
            ("system", CONDENSE_TEMPLATE),
            
        ])

        # Construct a conversational chain by combining the prompt and OpenAI client
        self.chain_qa = self.prompt_qa | self.model
        self.chain_condense = self.prompt_condense | self.model
        
    def chat(self, user_input):
        condense_question = self.chain_condense.invoke({
            "question": user_input,
            "chat_history": self.history
        })

        vectorstore_knowledge = PineconeVectorStore(
            index_name=self.index_name, embedding=self.embed_model, namespace=os.getenv("PINECONE_KNOWLEDGE_NAMESPACE")
        )

        query = condense_question.content
        
        print(f'=============condense query: {query}')

        docs_knowledge = vectorstore_knowledge.similarity_search(
            query,  # our search query
            k=20  # return 3 most relevant docs
        )

        separator = '\n\n'
        serialized_docs_knowledge = [doc.page_content for doc in docs_knowledge]

        knowledge_context = separator.join(serialized_docs_knowledge)


        separator = '\n'
        serialized_history = [f"{key}: {value}" for message in self.history[-5:] for key, value in message.items()]
        history = separator.join(serialized_history)
        print(f'=============history: {history}')
        print(f'=============knowledge_context: {knowledge_context}')
        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Convert the datetime object to a string
        current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                
        answer = self.chain_qa.invoke({
            "question": user_input,
            "chat_history": history,
            "context": knowledge_context,
            "current_time": current_datetime_str
        })
        self.history.append({"Human": user_input})
        self.history.append({"Assistant": answer.content})
        return answer.content
# Function to generate a unique UUID based on host ID and current time


