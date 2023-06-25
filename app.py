import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# load agents and tools modules
import pandas as pd
from io import StringIO
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import LLMMathChain

from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

# The easy document loader for text
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import CSVLoader

from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = 'Enter your api key here.'

model_name = 'text-embedding-ada-002'

# initialize embeddings object; for use with user query/input
embed = OpenAIEmbeddings(
                model=model_name
            )


loader = TextLoader('hr_policy.txt')
doc = loader.load()
print (f"You have {len(doc)} document")
print (f"You have {len(doc[0].page_content)} characters in that document")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
docs = text_splitter.split_documents(doc)
# print("---------------------------------------------------------------------------")
# print(docs[0])
# print("---------------------------------------------------------------------------")
# print(docs[1])
# print("---------------------------------------------------------------------------")
# print(docs[2])


# initialize langchain vectorstore
vectorstore = FAISS.from_documents(docs, embed)

# Get the total number of characters so we can see the average later
num_total_characters = sum([len(x.page_content) for x in docs])
# print([print(x) for x in docs])

print (f"Now you have {len(docs)} documents that have an average of {num_total_characters / len(docs):,.0f} characters ")


# initialize LLM object
llm = ChatOpenAI(    
    model_name="gpt-3.5-turbo", 
    )

# initialize vectorstore retriever object
timekeeping_policy = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

file = "employee_data.csv"
# csv_file = StringIO("employee_data.csv") 
df = pd.read_csv(file) # load employee_data.csv as dataframe
# print(df)
python = PythonAstREPLTool(locals={"df": df}) # set access of python_repl tool to the dataframe


# create calculator tool
calculator = LLMMathChain.from_llm(llm=llm, verbose=True)

# create variables for f strings embedded in the prompts
user = 'Alexander Verdad' # set user
df_columns = df.columns.to_list() # print column names of df
print(df_columns)

# prep the (tk policy) vectordb retriever, the python_repl(with df access) and langchain calculator as tools for the agent
tools = [
    Tool(
        name = "Timekeeping Policies",
        func=timekeeping_policy.run,
        description="""
        Useful for when you need to answer questions about employee timekeeping policies.
        """
    ),
    Tool(
        name = "Employee Data",
        func=python.run,
        description = f"""
        Useful for when you need to answer questions about employee data stored in pandas dataframe 'df'. 
        Run python pandas operations on 'df' to help you get the right answer.
        'df' has the following columns: {df_columns}
        
        <user>: How many Sick Leave do I have left?
        <assistant>: df[df['name'] == '{user}']['sick_leave']
        <assistant>: You have n sick leaves left.              
        """
    ),
    Tool(
        name = "Calculator",
        func=calculator.run,
        description = f"""
        Useful when you need to do math operations or arithmetic.
        """
    )
]

memory = ConversationBufferMemory(memory_key="chat_history")
# change the value of the prefix argument in the initialize_agent function. This will overwrite the default prompt template of the zero shot agent type
agent_kwargs = {'prefix': f'You are friendly HR assistant. You are tasked to assist the current user: {user} on questions related to HR. You have access to the following tools:'}


# agent = initialize_agent(tools, 
#                             llm,
#                             agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
#                             verbose=True, 
#                             memory=memory,
#                             agent_kwargs=agent_kwargs)
# initialize the LLM agent
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True, 
                         agent_kwargs=agent_kwargs
                         )


# define q and a function for frontend
def get_response(user_input):
    response = agent.run(user_input)
    print(response)
    return response

print(agent.agent.llm_chain.prompt.template)

# get_response('What is my name and employee id?')














