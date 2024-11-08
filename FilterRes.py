
import uuid


from langchain.tools import  tool
from pprint import pprint

from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import sqlite3

from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict
from typing import List
import os
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.prompts import PromptTemplate
#from raptor_feynman import answer_raptor



### Set the API keys
os.environ['GROQ_API_KEY'] 




###Choose LLM model and provider



llm = ChatGroq(
            model="llama-3.2-90b-vision-preview",
            temperature=0,
        )


###Prompt templates


#Unite final answer

prompt_steps_to_solve = PromptTemplate(
    template=
    """
    You are a helpful teacher that is trying to help a student solve a science questions from children from low-income families. \n
    Your job is to give the student a helpful answer about their doubts. \n
    Leave any extras that might help the user understand their question like examples or applications. \n
    Return the steps to solve the problem. \n
    Here is the problem: \n
    {problem} \n
    """,
   inputs=["problem"],
)


prompt_multi_question = PromptTemplate(
    template=
    """
    You are a helpful teacher that is trying to help kids from below 10 years to solve science questions about the world. \n
    Your job is to analyze the steps to solve the the question. \n
    Make at least three questions that will help the student understand the problem. \n
    Return a JSON key "questions" with the list of questions. \n
    Here is the problem: \n
    {problem} \n
    Here is the steps to solve the problem: \n
    {steps_to_solve} \n

    """,
   inputs=["problem" ,"steps_to_solve"],
)




prompt_solve_physics_problem = PromptTemplate(
    template=
    """
    You are a helpful teacher that is trying to help kids from below 10 years to solve science questions about the world. \n
    Your task is to use the information provided and format it in a way that the student can understand. \n
    Just return the explanation. \n
    Here is the problem: \n
    {problem} \n
    Vector data base information: {vector_data_base_answer} \n
    Steps to solve the problem: {steps_to_solve} \n
    """,
   inputs=["problem" ,  "vector_data_base_answer", "steps_to_solve"],
)

prompt_explain_problem = PromptTemplate(
    template=
    """
    You are a helpful teacher that is trying to help kids from below 10 years to solve science questions about the world. \n
    Your job is to modify the answer to make it easier to understand. \n
    Use ALWAYS emojis to the answer. \n
    The answer is for 1st grade students. \n
    Always answer in spanish. \n
    Not a superlarge answer. \n
    Use ALWAYS emojis to the answer. This emojis has to be related to the answer. \n
    Here are some more recommendations: \n
    From now on, you will always answer by telling me in detail how physical phenomena work. The questions will be asked by elementary school children, so they don't have much understanding of many words, so you have to use practical, fun examples, and for example, if someone asks you why there are different colors in the rainbow? You have to use an example of little balls, which resemble particles, and how these particles, for example, it's a very vague explanation, but they collide with water drops and depending on the angle that is formed, they will disperse one color or another. You have to explain a little what the angle is and so, it has to be a concise answer, not so simple, that is, more than simple, it must be very well structured, well told, well founded, and in such a way that I can really almost answer my doubt with just one question, beyond anything else, the user can continue asking you about specific questions, but you do have to be very very clear.
    Here is the original question: \n
    {question} \n
    Here is the answer to transform: \n
    {final_answer} \n
    """,
   inputs=["question","final_answer" ],
)


prompt_answer_question = PromptTemplate(
    template=
    """
    You are a helpful teacher that is trying to help kids from below 10 years to solve science questions about the world. \n
    Your job is to answer the question. \n
    Use ALWAYS emojis to the answer. \n
    Just return the answer. \n
    Here is the question: \n
    {question} \n
    """,
   inputs=["question"],
)



#Chains

chain_multi_question = prompt_multi_question | llm | JsonOutputParser()

chain_steps_to_solve = prompt_steps_to_solve | llm | StrOutputParser()

chain_solve_physics_problem = prompt_solve_physics_problem | llm | StrOutputParser()

chain_translate_problem = prompt_explain_problem | llm | StrOutputParser()

chain_answer_question = prompt_answer_question | llm | StrOutputParser()


#Graph State

class GraphState(TypedDict):
    problem: str


    steps: str
    questions: List[str]
    documents: str
    final_answer: str
    translate: str



#Utility functions

def multi_question(state):
    """
    Extract the multi question

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the multi question
    """

    problem = state["problem"]
    steps_to_solve = state["steps"]
    questions = chain_multi_question.invoke({"problem": problem, "steps_to_solve": steps_to_solve})
    print(questions)
    return {"questions": questions["questions"]}



def steps_to_solve(state):
    """
    Extract the steps to solve the problem

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the steps to solve the problem
    """

    problem = state["problem"]
    steps = chain_steps_to_solve.invoke({"problem": problem})
    return {"steps": steps} 


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    questions = state["questions"]
    documents = ""
    for question in questions:
        print(question)
        answer = chain_answer_question.invoke({"question": question})
        answer = str(answer)
        documents= documents + answer
        print(documents)
    # Retrieval

    return {"documents": documents}



def solve_physics_problem(state):
    """
    Analyze all the information provided and return a final answer.

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the final answer
    """

    problem = state["problem"]
    vector_data_base_answer = state["documents"]
    steps = state["steps"]
    final_answer = chain_solve_physics_problem.invoke({"problem": problem, "vector_data_base_answer": vector_data_base_answer, "steps_to_solve": steps})
    print(final_answer)
    return {"final_answer": final_answer}




def translate(state):
    """
    Transform the answer to make it easier to understand.

    Args:
    state(dict): current state of the graph

    Returns:

    state(dict): updated state of the graph with the translation
    """
    question = state["problem"]
    final_answer = state["final_answer"]
    translation = chain_translate_problem.invoke({"final_answer": final_answer, "question": question})
    print(translation)
    return {"translate": translation}


#Graph
workflow_filter = StateGraph(GraphState)


workflow_filter.add_node("steps_to_solve", steps_to_solve)

workflow_filter.add_node("multi_question", multi_question)

workflow_filter.add_node("vector_database_answer", retrieve)


workflow_filter.add_node("solve_physics_problem",solve_physics_problem)

workflow_filter.add_node("translate_problem", translate)


workflow_filter.set_entry_point("steps_to_solve")

workflow_filter.add_edge("steps_to_solve", "multi_question")

workflow_filter.add_edge("multi_question", "vector_database_answer")

workflow_filter.add_edge("vector_database_answer",  "solve_physics_problem")

workflow_filter.add_edge("solve_physics_problem", "translate_problem")

workflow_filter.add_edge("translate_problem", END)




def from_conn_stringx(cls, conn_string: str,) -> "SqliteSaver":
    return SqliteSaver(conn=sqlite3.connect(conn_string, check_same_thread=False))
#Memory
SqliteSaver.from_conn_stringx=classmethod(from_conn_stringx)

SqliteSaver.from_conn_stringx=classmethod(from_conn_stringx)

memory = SqliteSaver.from_conn_stringx(":memory:")



# Compile
app = workflow_filter.compile(checkpointer=memory)

#config

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "patient_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}


    

def run_workflow_filter(inputs):
    for output in app.stream(inputs,config):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")
    pprint(value["translate"])
    return value["translate"]


run_workflow_filter({"problem": "Porque vemos colores?"})
