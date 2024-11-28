import os
import random
import base64
import chardet
# import sqlite3
import aiosqlite
from yaml import safe_load
from PIL import Image as PILImage
from langchain_groq import ChatGroq
from langchain_pinecone import Pinecone
from langchain_core.messages import HumanMessage
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.retrievers import SelfQueryRetriever
from src.agents.utils import create_member_agent_node
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from src.agents.supervisor import create_supervisor_node
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits import create_sql_agent
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

def create_client_manifest_path(client_name, manifest_file_path):
    """
    Create the path for the client's manifest file.

    Args:
        client_name (str): The name of the client.
        manifest_file_path (str): The original manifest file path.

    Returns:
        str: The new manifest file path.
    """
    directory, filename = os.path.split(manifest_file_path)
    new_directory = os.path.join(directory, client_name)
    return os.path.join(new_directory, filename)

def load_config(filepath):
    """
    Load the configuration from a YAML file.

    Args:
        filepath (str): Path to the YAML file.

    Returns:
        dict: The loaded configuration.
    """
    with open(filepath, 'r') as file:
        return safe_load(file)


def get_product_information(file_path):
    with open(file_path, 'r') as file:
        product_information = safe_load(file)
    return product_information['product_information']


def extract_intents_objectives_sales_steps(file_path):
    with open(file_path, 'r') as file:
        data = safe_load(file)   
    return data

def get_active_client_info(config):
    """
    Get the information of the active client from the configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        dict: The active client's information.
    """
    for client in config['clients']:
        if client['status'] == 'active':
            return client
    return None

def initialize_models(config, client_name):
    """
    Initialize the models and tools based on the configuration.

    Args:
        config (dict): The configuration dictionary.
        client_name (str): The name of the active client.

    Returns:
        tuple: Initialized model, embeddings model, and memory.
    """
    agent_model_name                  = config['model']['agent_model']
    response_refining_model_name      = config['model']['response_refining_model']
    quick_response_model_name         = config['model']['quick_response_model']

    chatbot_responses_model_map = {
                                    'quick_response'            : quick_response_model_name,
                                    'bot_response'              : agent_model_name,
                                    'refined_response'          : response_refining_model_name
                                }
    
    temperature                       = config['model']['randomness']
    quick_response_model_temperature  = config['model']['quick_response_model_randomness']
    embeddings_model_name             = config['embeddings_model']['name']
    
    model                             = ChatOpenAI(model=agent_model_name, temperature=temperature)
    quick_response_model              = ChatGroq(model=quick_response_model_name, temperature=quick_response_model_temperature, max_tokens=5)
    response_refining_model           = ChatOpenAI(model=response_refining_model_name, temperature=temperature)

    embeddings_model                  = OpenAIEmbeddings(model=embeddings_model_name)
    conn                              = aiosqlite.connect(":memory:")#, check_same_thread=False)  
    memory                            = AsyncSqliteSaver(conn)
    
    return model, embeddings_model, memory, quick_response_model, response_refining_model, chatbot_responses_model_map


def inject_product_info_into_tool(model, tool_info, product_information, use_case):
    """
    Injects product-specific information into the provided tool description using an LLM, dynamically handling any number of attributes.

    Args:
        model: The language model to process the information.
        tool_info (dict): The original tool description that needs updating. {dict contains name and description}
        product_information (dict): A dictionary containing the new product details like attributes.

    Returns:
        str: The updated tool description with the product-specific information injected.
    """

    def _parse(text):
        """Helper function to clean and parse model output."""
        return text.strip('"').strip("**")

    if use_case == 'tool_description':
        template = """
        You are tasked with updating the following tool description for a chatbot that assists with {product}. 
        Replace any references related to the previous product (e.g., wine) with the information for the new product. 
        Do not depend on the number of attributes for previous product wine which is 5. The new product can have any number of attributes, make sure to use all the information of new product.

        Product Information:
        {product_information}

        Tool Name: {tool_name} 
        Tool Description:
        {tool_data}

        Please update the tool description based on the new product information and return only the updated description.
        """
    if use_case == 'tool_prompt':
        template = """
        You are tasked with updating the following tool prompt for a chatbot that assists with {product}. 
        Replace any references related to the previous product (e.g., wine) with the information for the new product. 
        The new product can have any number of attributes, make sure to use all the information of the new product.
        Utilize complete product information when updating prompt template, leave placeholder that are not present in product information.

        Product Information:
        {product_information}

        Prompt Name: {tool_name}
        Prompt Template:
        {tool_data}

        Please update the prompt template based on the new product information and return only the updated prompt. Do not return anything else.

        Updated Prompt Template:
        """


    prompt = ChatPromptTemplate.from_template(template)

    print("product_information : \n",product_information)
    chain = prompt | model | StrOutputParser() | _parse

    updated_tool_data = chain.invoke({
            "product": product_information['product'],
            "product_information": product_information,
            "tool_name": tool_info["name"],
            "tool_data": tool_info["tool_data"]
        })

    return updated_tool_data

async def generate_external_input(original_user_query: str, conversation_history: str) -> str:
    def _parse(text):
        return text.strip('"').strip("**")

    template = """
    Analyze the following user message and conversation history to provide insights about the customer's behavior, emotional state, or preferences that can improve customer experience and wine recommendation:

    User message: {original_user_query}

    Conversation history:
    {conversation_history}

    Provide a one line external input that captures important observations about the customer. This may include their emotional state, level of engagement, potential frustrations, or any other relevant insights. Keep it concise and focused on the most important observations.
    If Conversation history does not exist, return 'no observation'. 

    External Input:
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model_name="gpt-4o-mini")
    chain = prompt | model | StrOutputParser() | _parse

    external_input = await chain.ainvoke({
        "original_user_query": original_user_query,
        "conversation_history": conversation_history
    })

    return external_input

async def generate_quick_response(query, model):
    prompt_template = (
        f"Given the user's query: '{query}', first check if it is a greeting like 'hello', 'hi', 'hey', or similar. "
        f"If the query is a greeting, respond with only an emoji such as '👋'. "
        f"If the query is not a greeting, provide a brief and contextually relevant acknowledgment of the user's intent. "
        f"The acknowledgment should be very short, concise, varied, and tailored to the user's query, indicating understanding and readiness to assist, without providing a full answer. "
        f"The response should be 2-3 words only, demonstrating understanding or enthusiasm."
    )
    messages = [
        (
            "system",
            "You are a helpful assistant that provides short, positive acknowledgments based on the user's intent. "
            "For greetings like 'hello', 'hi', 'hey', respond only with an emoji. Otherwise, respond with a concise acknowledgment."
        ),
        ("human", prompt_template)
    ]
    llm = ChatGroq(model="gemma2-9b-it", temperature=0.0)
    ai_msg = await llm.ainvoke(messages)
    quick_response = ai_msg.content.strip()

    return quick_response


async def refine_final_response(response, model):
    def _parse(text):
        return text.strip('"').strip("**")
        

    template = """Refine the following response to sound more natural and conversational. Add filler words (like "um," "uh," "well," or similar) and natural pauses ("...", ",") to mimic human speech patterns. Include 2 to 3 filler words in total, spread throughout the response, to enhance engagement without overusing them. Ensure the refined response retains the original intent while incorporating these elements naturally.

    **Original Response:**
    {response}

    **Refined Response:**
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser() | _parse
    refined_response = await chain.ainvoke({"response": response})
   
    return refined_response

async def create_input_message_from_user(model, history, query="", frontend=False):
    """
    Create an input message from user input.

    Returns:
        tuple: A tuple containing the input message and the user's query.
    """
    if frontend:
        query = query
    else:
        query = input("user : ")
    
    refined_query = await contextual_query_refiner(model, query, history)
    print("refined_query: ", refined_query)
    input_message = [HumanMessage(content=refined_query)]

    return input_message, query
    
def initialize_vectorstore(config, embeddings_model):
    """
    Initialize the vectorstore using Pinecone.

    Args:
        config (dict): The configuration dictionary.
        embeddings_model (OpenAIEmbeddings): The initialized embeddings model.

    Returns:
        Pinecone: The initialized vectorstore.
    """
    index_name = config['pinecone']['index_name']
    namespace = config['pinecone']['namespace']
    
    return Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings_model,
        namespace=namespace,
    )

def initialize_retriever(model, vectorstore, config):
    """
    Initialize the self-query retriever.

    Args:
        model (ChatOpenAI): The initialized model.
        vectorstore (Pinecone): The initialized vectorstore.
        config (dict): The configuration dictionary.

    Returns:
        SelfQueryRetriever: The initialized self-query retriever.
    """
    document_content_description = config['pinecone']['document_content_description']
    
    metadata_field_info = [
        AttributeInfo(
            name="liquor_style",
            description="The style of the wine",
            type="string or list[string]",
        ),
        AttributeInfo(
            name="variety",
            description="The variety of the wine",
            type="string",
        ),
        AttributeInfo(
            name="price",
            description="The price of the wine in between ranges 10-10000",
            type="integer",
        ),
    ]
    
    min_docs = 2
    max_docs = 5
    num_docs_to_retrieve = random.randint(min_docs, max_docs)


    return SelfQueryRetriever.from_llm(
        model,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_kwargs={"k": num_docs_to_retrieve},
    )

def setup_database(host, username, password, port, database):
    """
    Setup the database from the client's CSV file.

    Args:
        host (str): The database host.
        username (str): The database username.
        password (str): The database password.
        port (str): The database port.
        database (str): The database name.

    Returns:
        SQLDatabase: The initialized SQL database.
    """
    db = SQLDatabase.from_uri(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}")
    return db


def initialize_sql_agent(model, sql_db):
    """
    Initialize the SQL agent with the given model and SQL database without schema checks.


    Args:
        model (ChatOpenAI): The initialized model.
        sql_db (SQLDatabase): The initialized SQL database.

    Returns:
        SQLAgent: The initialized SQL agent.
    """

    llm= ChatOpenAI(model= "gpt-4o-mini", temperature= 0.0)
    #schema_description = sql_db.get_table_info()
    sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)

    MSSQL_AGENT_FORMAT_INSTRUCTIONS = """
    ## Use the following format:
    Example of final answer:
    Price : $22
    
    """

    return create_sql_agent(
        llm=llm,
        format_instructions=MSSQL_AGENT_FORMAT_INSTRUCTIONS,
        toolkit=sql_toolkit,
        agent_type="openai-tools",
        top_k=10,
        handling_errors=True,
        verbose=True,

    )

async def create_nodes(model, memory, agents_prompt, tools_per_agent, reasoning_flag, supervisor_agent_name='Supervisor'):
    """
    Create nodes for agents based on the given prompts and tools.

    Args:
        agents_prompt (dict): Dictionary containing prompts for agents.
        tools_per_agent (dict): Dictionary containing tools for agents.
        supervisor_agent_name (str): Name of the supervisor agent. Defaults to 'Supervisor'.

    Returns:
        dict: Dictionary of created agent nodes.
    """
    final_agents_nodes = {}
    members_name_list = []

    # Collect names of all members excluding the supervisor
    for key, value in agents_prompt.items():
        if value['agent_name'] != supervisor_agent_name:
            members_name_list.append(f"{value['agent_name']}_node")
    print("\n\t [INFO]: Final Team Members : ", members_name_list)

    # Create nodes for each agent based on their prompt and tools
    for key, value in agents_prompt.items():
        agent_name = value['agent_name']
        node_name = f"{agent_name}_node"
        agent_system_prompt = value['agent_system_prompt']
        agent_tools = [tool_function for tool_name, tool_function in tools_per_agent.get(agent_name, {}).items()]

        for tool in agent_tools:
            print(tool, '\n')

        if node_name in members_name_list:
            print("\n [INFO]: Creating Node - ", node_name)
            member_agent_node = await create_member_agent_node(model, node_name, agent_system_prompt, agent_tools, memory, reasoning_flag)
            final_agents_nodes[node_name] = member_agent_node
        else:
            print("\n [INFO]: Creating Supervisor Node - ", node_name)
            supervisor_agent_node = create_supervisor_node(model, agent_system_prompt, members_name_list)
            final_agents_nodes[node_name] = supervisor_agent_node

    print("\n\t [INFO]: Final Agent Nodes : ", final_agents_nodes.keys())
    return final_agents_nodes

def show_workflow_graph(graph):
    """
    Display the workflow graph by saving it as an image and opening it.

    Args:
        graph: The workflow graph object to be displayed.
    """
    image_data = graph.get_graph().draw_png()
    with open('./graph_image.png', 'wb') as f:
        f.write(image_data)
    img = PILImage.open('./graph_image.png')
    img.show()

async def get_history_from_graph(graph, config):
    history = []
    graph_state = await graph.aget_state(config)
    messages = graph_state.values["messages"]
    messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    for i, message in enumerate(messages):
        prefix = "user: " if i % 2 == 0 else "ai: "
        history.append(prefix + message.content)
    history = "\n".join(history)
    return history
    
async def contextual_query_refiner(model, query, history):
    def _parse(text):
        return text.strip('"').strip("**")
    
    # Check if the history is empty
    if not history.strip():

        return query

    # Extract last message from the history
    last_message = history.splitlines()[-1]
    template = f""" 
            1. If user is asking a follow-up question based on a previous conversation. Refine the user's query by adding the specific topic discussed in the last message. Ensure that the refined query aligns with the context of the last message, focusing only on that topic. But DO NOT add additional information which is not in previous messages & original query.
            2. If the user's original query indicates a shift in focus or a new direction compared to the previous messages, 
	disregard the outdated context and craft the refined query to align with the user's new intent.
	        3. If the user's original query is simple or a straightforward confirmation, retain the query in its original form without any modifications or elaboration.

            Important: 
            - ALWAYS remember DO NOT add additional information which is not in previous messages & original query.
            - Return only the refined query as output without any additional explanations or context.
            - The refined query should be precise, contextually relevant, and aligned with the user's current intent. 
            - If no previous messages exist, retain the query in its original form.

            Previous messages:
            {history}
            
            Last message:
            {last_message}
        
            Original user query:
   
            {query}
        
            Refined query:
                """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser() | _parse
    refined_query = await chain.ainvoke({"query": query, "history": history})
    
    return refined_query
    
def create_input_message_from_image(image_path='./detected_frame.jpg'):
    """
    Create an input message containing an image encoded in base64.

    Args:
        image_path (str): Path to the image file. Defaults to './detected_frame.jpg'.

    Returns:
        list: List of AI and Human messages with the image.
    """
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # system_prompt, _ = generate_system_prompt("./intents.yaml", "greet")
    # print("[INFO]:", system_prompt)
    
    pattern = r"Task: (.*?)Instructions:"
    # match = re.search(pattern, system_prompt, re.DOTALL)
    # task_section = match.group(1).strip() if match else "Task not found"
    # print("[INFO]:", task_section)
    
    input_message = [
        # AIMessage(content=system_prompt),
        HumanMessage(content=[
            # {"type": "text", "text": task_section},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])
    ]
    return input_message

def detect_encoding(file_path):
    """
    Detect the encoding of a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Detected encoding of the file.
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def format_background_info(background_info):
    formatted_info = "\n".join(
        f"{key.replace('_', ' ').title()}:\n{value}"
        for key, value in background_info.items() if key != 'magic_prompt'
    )
    return f"{background_info['magic_prompt']}\n\n{formatted_info}"
