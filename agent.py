from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_google_genai import ChatGoogleGenerativeAI


class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = PyPDFLoader(file_path)
        docs = self.loader.load()
        
        self.embedding = GoogleGenerativeAIEmbeddings(
            google_api_key="AIzaSyDmvQjyWNCDVo1f1GOX41ip7LVe9FXcRNc",
            model="models/embedding-001",
        )

        self.initDB()
        self.vectorstore.add_documents(docs)
    
    def initDB(self):
        self.vectorstore = Chroma(
            collection_name="pdf_embeddings",
            embedding_function=self.embedding,
            persist_directory="pdf_embeddings",
        )



llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    google_api_key="AIzaSyDmvQjyWNCDVo1f1GOX41ip7LVe9FXcRNc",
)

class AgentState(TypedDict):
    question: str
    query: str
    output: str
    final_answer: str

class Agent:
    def __init__(self, model, vdb):
        self.model = model
        self.vdb = vdb
        self.graph = StateGraph(AgentState)
        self.graph.set_entry_point("start")
        self.graph.add_node("start",self.start)
        self.graph.add_node("get_query", self.query_vdb)
        self.graph.add_node("get_output", self.llm_answer)

        self.graph.add_edge("start", "get_query")
        self.graph.add_edge("get_query", "get_output")
        self.graph.add_edge("get_output",END)
        
        self.graph = self.graph.compile()
    
    def start(self,state:AgentState):
        # print("On Start")
        question = state['question']
        prompt = ([HumanMessage(f"""
        You are an helpfull assistant you will be provided a question. You have to create a query for the VDB

        You will be provided questions like "What frameworks are you familiar with?". You have to craft a suitable query for a VDB containing the Resume.

        The output must be only the query do not add any additional messages only the query. The query must be suitable to a Vector Databse so the query must be in english
        
        The question:
        {question}

        """)])
        result = self.model.invoke(prompt)
        # print(result.content) #prompt being forwarded to model which is the LLM, then response stored in the result
        state["query"] = result #result response is then stored to the state
        return {"query": result.content} #state values are returned
    def query_vdb(self, state):
        # print("On get Qury")
        query = state['query']
        result = self.vdb.similarity_search(query)
        # print(result)
        return {'output': result}
    
    def llm_answer(self, state):
        # print("On get out")

        question = state['question']
        output = state['output']

        prompt = f"""
        I will provde you with a question about a resume and an details regarding that resume. Your task is to answer that question based on the resume.

        The Question:
        {question}

        The Deatils:
        {output}


        The output must be the answer to the question based on given details
        
        """
        prompt = ([HumanMessage(prompt)

        ])
        

        result = self.model.invoke(prompt)
        print(result.content)
        return {"final_answer": result.content}

    
def main():
        pdf_loader = PDFLoader("Resume.pdf")
        vectorstore = pdf_loader.vectorstore
        agent = Agent(llm, vectorstore) 

        while True:
            question = input("Ask a question (or type 'exit'): ")
            if question.lower() == "exit":
                break
            result = agent.graph.invoke({"question": question})
            print("Answer:", result['final_answer'])


main()




