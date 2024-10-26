import os
from typing import Annotated, Sequence, TypedDict, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, Graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API Key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize models
model = ChatOpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

# Create vector database from knowledge base
def create_vector_db(knowledge_file: str):
    """Create FAISS vector database from knowledge base file"""
    try:
        with open(knowledge_file, 'r') as f:
            raw_text = f.read()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(raw_text)
        
        print(f"Created {len(chunks)} chunks from knowledge base")
        
        # Create vector store
        vector_store = FAISS.from_texts(chunks, embeddings)
        return vector_store
    except FileNotFoundError:
        print(f"Error: Knowledge base file '{knowledge_file}' not found!")
        exit(1)

# Initialize vector store
print("Initializing vector store from knowledge base...")
vector_store = create_vector_db('./airtel_knowledge.txt')

# Define state structure
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    current_agent: str
    user_details: dict
    intent: str

def log_state(state: AgentState, stage: str):
    """Helper function to log the state at any stage for debugging."""
    print(f"\n[DEBUG - {stage}] State Information:")
    print(f"Intent: {state['intent']}")
    print(f"Current Agent: {state['current_agent']}")
    print(f"User Details: {state['user_details']}")
    print(f"Messages: {[msg.content for msg in state['messages']]}")
    print("===")

def router_agent(state: AgentState) -> AgentState:
    """Routes the query to appropriate agent based on intent"""
    
    log_state(state, "Before Routing")
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intent classifier for Airtel customer service.
        Classify the user query into one of these categories:
        - general_query: For general questions about Airtel services
        - sim_swap: If user wants to swap/replace their SIM
        - plan_details: If user wants to know their plan details
        - conversation: For basic conversational messages like greetings, thanks, acknowledgments
        - fallback: For specific tasks or questions unrelated to Airtel
        
        Examples:
        - "thanks" → conversation
        - "ok" → conversation
        - "hello" → conversation
        - "what's the weather like?" → fallback
        - "can you write me a poem?" → fallback
        
        Respond with just the category name."""),
        ("user", "{input}")
    ])
    
    latest_message = state["messages"][-1].content
    response = model.invoke(router_prompt.format_messages(input=latest_message))
    intent = response.content.lower().strip()

    # Keep the intent as sim_swap if we're already in the sim swap flow
    if state["intent"] == "sim_swap" and (
        state["messages"][-2].content == "To process your SIM swap request, I'll need some details. Please provide your phone number:" or 
        state["messages"][-2].content == "Thank you. Now, please provide your full name:"
    ):
        intent = "sim_swap"
    
    # Keep plan_details intent if we're collecting phone number
    if state["intent"] == "plan_details" and state["messages"][-2].content == "To check your plan details, please provide your phone number:":
        intent = "plan_details"

    state["intent"] = intent
    state["current_agent"] = intent
    log_state(state, "After Routing")
    return state

def conversation_agent(state: AgentState) -> AgentState:
    """Handles basic conversational messages"""
    
    conversation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an Airtel customer service bot. 
        Respond naturally to conversational messages while staying in character.
        Keep responses brief and friendly.
        If appropriate, remind the user you can help with Airtel services.
        
        Examples:
        - For "thanks": "You're welcome! Let me know if you need any other assistance with Airtel services."
        - For "ok": "Great! I'm here if you need any help with Airtel services."
        - For "hello": "Hello! How can I assist you with Airtel services today?"
        """),
        ("human", "{message}")
    ])
    
    latest_message = state["messages"][-1].content
    response = model.invoke(conversation_prompt.format_messages(message=latest_message))
    
    state["messages"].append(AIMessage(content=response.content))
    return state

def general_query_agent(state: AgentState) -> AgentState:
    """Handles general queries about Airtel services using RAG"""
    latest_message = state["messages"][-1].content
    
    # Search relevant context from vector store
    relevant_chunks = vector_store.similarity_search(latest_message, k=3)
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])
    
    # Create prompt with context
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an Airtel customer service bot. Use the following context to answer the question.
        If the answer cannot be found in the context, only use well-known general information about Airtel.
        Keep responses concise and helpful.
        
        Context:
        {context}"""),
        ("human", "{question}")
    ])
    
    # Generate response using context
    response = model.invoke(
        rag_prompt.format_messages(
            context=context,
            question=latest_message
        )
    )
    
    state["messages"].append(AIMessage(content=response.content))
    return state

def sim_swap_agent(state: AgentState) -> AgentState:
    """Handles SIM swap requests"""
    
    if "phone_number" not in state["user_details"]:
        response = "To process your SIM swap request, I'll need some details. Please provide your phone number:"
        state["messages"].append(AIMessage(content=response))
        return state
    
    if "name" not in state["user_details"]:
        # Only ask for name if we just collected the phone number
        if state["messages"][-1].content.strip().isdigit():
            response = "Thank you. Now, please provide your full name:"
            state["messages"].append(AIMessage(content=response))
            return state
        return state  # Let collect_details handle the name collection
    
    # Show final confirmation
    response = f"""Thank you for providing your details:
    Name: {state['user_details']['name']}
    Phone: {state['user_details']['phone_number']}
    
    Your SIM swap request has been registered. Please visit your nearest Airtel store with valid ID proof to complete the process."""
    
    state["messages"].append(AIMessage(content=response))
    return state

def plan_details_agent(state: AgentState) -> AgentState:
    """Handles plan details queries"""
    
    if "phone_number" not in state["user_details"]:
        response = "To check your plan details, please provide your phone number:"
        state["messages"].append(AIMessage(content=response))
        return state
    
    # Mock plan details
    standard_plan = """Your current active plan details:
    Plan: Airtel Unlimited
    Data: 2GB/day
    Validity: 84 days
    Calls: Unlimited
    SMS: 100/day"""
    
    state["messages"].append(AIMessage(content=standard_plan))
    return state

def fallback_agent(state: AgentState) -> AgentState:
    """Handles unrelated queries"""
    
    response = """I apologize, but I can only assist with Airtel-related queries. 
    I can help you with:
    - General information about Airtel services
    - SIM swap requests
    - Checking your plan details
    
    Please let me know if you need help with any of these."""
    
    state["messages"].append(AIMessage(content=response))
    return state

def should_collect_details(state: AgentState) -> str:
    """Determines if we need to collect user details for SIM swap and plan details workflows."""
    current_intent = state["intent"]
    latest_message = state["messages"][-1].content

    if current_intent == "conversation":
        return "conversation"
    
    if current_intent == "sim_swap":
        if "phone_number" not in state["user_details"]:
            # Check if current message is a phone number
            if latest_message.strip().isdigit():
                return "collect_details"
            return "sim_swap"  # Ask for phone number
            
        if "name" not in state["user_details"]:
            # Check if we just got the phone number
            if latest_message.strip().isdigit():
                return "sim_swap"  # Ask for name
            return "collect_details"  # Collect the name
            
        return "sim_swap"  # Process the final confirmation
    
    # For plan details flow
    if current_intent == "plan_details":
        if "phone_number" not in state["user_details"]:
            if latest_message.strip().isdigit():
                return "collect_details"
            return "plan_details"  # Ask for phone number
        return "plan_details"  # Show plan details

    # For other intents
    return current_intent or "fallback"

def collect_user_details(state: AgentState) -> AgentState:
    """Collects user details for sim swap and plan details workflows"""
    latest_message = state["messages"][-1].content.strip()
    current_intent = state["intent"]

    # Handle phone number collection
    if "phone_number" not in state["user_details"] and latest_message.isdigit():
        state["user_details"]["phone_number"] = latest_message
        return state

    # Handle name collection for sim swap
    if (current_intent == "sim_swap" and 
        "phone_number" in state["user_details"] and 
        "name" not in state["user_details"] and 
        not latest_message.isdigit()):
        state["user_details"]["name"] = latest_message
        return state

    return state

def get_next_step(state: AgentState) -> str:
    """Determines the next step after collecting details."""
    current_intent = state["intent"]
    
    if current_intent == "sim_swap":
        # After collecting phone number, go to sim_swap to ask for name
        if "phone_number" in state["user_details"] and "name" not in state["user_details"]:
            return "sim_swap"
        # After collecting name, go to sim_swap for final confirmation
        if "phone_number" in state["user_details"] and "name" in state["user_details"]:
            return "sim_swap"
        return "collect_details"

    if current_intent == "plan_details":
        if "phone_number" in state["user_details"]:
            return "plan_details"
        return "collect_details"

    return current_intent or "fallback"


def chat_with_airtel(user_input: str, state: dict = None) -> dict:
    if state is None:
        state = {
            "messages": [HumanMessage(content=user_input)],
            "current_agent": "router",
            "user_details": {},
            "intent": ""
        }
    else:
        state["messages"].append(HumanMessage(content=user_input))
        
        # Keep the current intent for ongoing flows
        if not state["intent"] in ["sim_swap", "plan_details"]:
            state["intent"] = ""
            state["current_agent"] = "router"
    
    # Create workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_agent)
    workflow.add_node("general_query", general_query_agent)
    workflow.add_node("sim_swap", sim_swap_agent)
    workflow.add_node("plan_details", plan_details_agent)
    workflow.add_node("conversation", conversation_agent)
    workflow.add_node("fallback", fallback_agent)
    workflow.add_node("collect_details", collect_user_details)

    # Add conditional edges
    workflow.add_conditional_edges(
        "router",
        should_collect_details,
        {
            "general_query": "general_query",
            "sim_swap": "sim_swap",
            "plan_details": "plan_details",
            "conversation": "conversation",
            "fallback": "fallback",
            "collect_details": "collect_details"
        }
    )

    # Add conditional edges from collect_details
    workflow.add_conditional_edges(
        "collect_details",
        get_next_step,
        {
            "sim_swap": "sim_swap",
            "plan_details": "plan_details",
            "collect_details": "collect_details",
            "fallback": "fallback"
        }
    )

    # Set entry point
    workflow.set_entry_point("router")
    
    # Run the workflow
    result = workflow.compile().invoke(state)
    return result

def print_bot_response(state):
    """Helper function to print the last bot response"""
    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            print("\nAirtel Bot:", last_message.content)
        else:
            print("\nUser:", last_message.content)

def run_test_conversation():
    """Function to run interactive conversation with the bot"""
    print("Welcome to Airtel Customer Service Bot!")
    print("Type 'quit' to exit the conversation\n")
    
    state = None
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            print("\nThank you for using Airtel Customer Service. Goodbye!")
            break
            
        state = chat_with_airtel(user_input, state)
        print_bot_response(state)

def run_automated_tests():
    """Function to run automated test cases"""
    print("\n=== Running Automated Test Cases ===\n")
    
    # Test Case 1: RAG-based General Query
    print("Test Case 1: General Query about 5G")
    state = chat_with_airtel("Tell me about Airtel's 5G coverage")
    print_bot_response(state)
    print("\n---")
    
    # Test Case 2: RAG-based Plan Information
    print("\nTest Case 2: General Query about Plans")
    state = chat_with_airtel("What prepaid plans does Airtel offer?")
    print_bot_response(state)
    print("\n---")
    
    # Test Case 3: SIM Swap Flow
    print("\nTest Case 3: SIM Swap Flow")
    state = chat_with_airtel("I need to swap my SIM card")
    print_bot_response(state)
    state = chat_with_airtel("9876543210", state)
    print_bot_response(state)
    state = chat_with_airtel("John Doe", state)
    print_bot_response(state)
    print("\n---")
    
    # Test Case 4: Fallback Query
    print("\nTest Case 4: Fallback Query")
    state = chat_with_airtel("What's the weather like today?")
    print_bot_response(state)
    print("\n---")

    print("\nTest Case 5: Basic Conversation")
    state = chat_with_airtel("thank you")
    print_bot_response(state)
    state = chat_with_airtel("hello", None)  # New conversation
    print_bot_response(state)
    print("\n---")

if __name__ == "__main__":
    while True:
        print("\nAirtel Bot Testing Options:")
        print("1. Start Interactive Conversation")
        print("2. Run Automated Test Cases")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            run_test_conversation()
        elif choice == "2":
            run_automated_tests()
        elif choice == "3":
            print("\nExiting the program. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")
