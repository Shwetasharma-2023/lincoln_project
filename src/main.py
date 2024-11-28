from dotenv import load_dotenv
_ = load_dotenv()
from langchain_core.messages import HumanMessage
import os
import json
import time
import uuid
import logging
import numpy as np
import pandas as pd
from src.tools.Manifests.wine import tools
from src.utils import get_product_information, extract_intents_objectives_sales_steps
from src.agents.workflow import create_workflow
from src.config import product_information_filepath , intent_file_path
from src.tools.utils import read_agent_tools_description, get_agent_tools
from src.agents.utils import inject_product_information, generate_active_agents_prompt
from src.utils import (
    format_background_info,
    create_nodes,
    show_workflow_graph,
    get_history_from_graph,
    create_input_message_from_image,
    create_input_message_from_user,
    generate_external_input,
    generate_quick_response,
    refine_final_response
)
from src.config import (
    model,
    quick_response_model,
    response_refining_model,
    memory,
    supervisor_agent_name, 
    tool_manifests_file_path,
    agent_manifests_file_path,
    chatbot_responses_model_map,
    conversation_logs_dir
)
# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------- Product Information ----------------- #
product_information = get_product_information(product_information_filepath)
intent_data= extract_intents_objectives_sales_steps(intent_file_path)

# -------------------------------------------- Time Profiling Config-------------------------------------------- #

concatenated_filename_string = "_".join(chatbot_responses_model_map.values())

quick_response_time_profiling_label           = 'quick_response'      + '_' + chatbot_responses_model_map['quick_response']
reasoning_n_bot_response_time_profiling_label = 'bot_response'        + '_' + chatbot_responses_model_map['bot_response']
refined_response_time_profiling_label         = 'refined_response'    + '_' + chatbot_responses_model_map['refined_response']

time_stats = {
    quick_response_time_profiling_label: [],
    reasoning_n_bot_response_time_profiling_label: [],
    refined_response_time_profiling_label: []
}

async def average_time_taken(time_stats):
    averages = {step: round(np.mean(times), 2) for step, times in time_stats.items()}
    for step, avg_time in averages.items():
        logging.info(f"\tAverage time for {step}: {avg_time} seconds")
    
    return averages

async def save_to_csv(averages, session_id):
    time_profiling_csv_path = os.path.join(conversation_logs_dir, f'{session_id}.csv')

    averages['run_id'] = session_id
    columns = [
        "run_id", 
        quick_response_time_profiling_label, 
        reasoning_n_bot_response_time_profiling_label, 
        refined_response_time_profiling_label
    ]
    df_new = pd.DataFrame([averages], columns=columns)

    if os.path.exists(time_profiling_csv_path):
        df_existing = pd.read_csv(time_profiling_csv_path)
        if session_id in df_existing['run_id'].values:
            df_existing.loc[df_existing['run_id'] == session_id, columns[1:]] = df_new[columns[1:]].values
        else:
            df_existing = pd.concat([df_existing, df_new], ignore_index=True)
        df_existing.to_csv(time_profiling_csv_path, index=False)
    else:
        df_new.to_csv(time_profiling_csv_path, mode='w', index=False, header=True)

# -------------------------------------------------------------------------------------------------------------- #

def save_logs(log_data, session_id):
    output_path = os.path.join(conversation_logs_dir, f'{session_id}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)

async def initialize_workflow(reasoning_flag:bool, product_information:dict, intent_data: dict):
    """
    Initialize the workflow by creating nodes and setting up the graph.
    """
    supervisor_node_name = supervisor_agent_name + '_node'
    agents_prompt, background_information = generate_active_agents_prompt(agent_manifests_file_path, product_information, intent_data)
    agents_tool_information = read_agent_tools_description(model, tool_manifests_file_path, product_information)

    tools_per_agent = get_agent_tools(agents_tool_information, agents_prompt)

    graph_nodes = await create_nodes(model, memory, agents_prompt, tools_per_agent, reasoning_flag, supervisor_agent_name)
    workflow = await create_workflow(graph_nodes, supervisor_node_name, memory, reasoning_flag)
    
    #show_workflow_graph(workflow)
    
    return workflow, format_background_info(background_information)


agents_prompt, background_information = generate_active_agents_prompt(agent_manifests_file_path, product_information, intent_data )

def generate_message_id():
    return str(uuid.uuid4())

async def handle_user_input(query: str, graph, config, logs, reasoning_flag, session_id, frontend=False, ):

    final_response = None
    tools.metadata_list = []
    history = await get_history_from_graph(graph, config)

    # print("GRAPH History : ", history)
    
    input_message, query = await create_input_message_from_user(model, history, query, frontend)

    if query == "exit":
        yield  ("stop",0)
    
    #external_input = await generate_external_input(query, history)

    input_message = input_message[0].content 
    logging.info(f"\n\tInput Message : {input_message}")
    input_message = [HumanMessage(content=input_message)]
    message_id = generate_message_id()

    # ------------------------- Step 1: Quick Response ------------------------- #
    start_time = time.time()  
    quick_response = await generate_quick_response(query, quick_response_model)
    yield ("quick", quick_response)
    
    end_time = time.time()  
    time_taken = end_time - start_time
    time_stats[quick_response_time_profiling_label].append(time_taken)
    logging.info(f"Time taken for quick response: {time_taken:.4f} seconds")  
    
    # Step 2: Detailed Response
    start_time = time.time()
    async for event in graph.astream({"messages": input_message}, config=config, stream_mode="values"):
        if reasoning_flag:
            if "reasoning_output" in event.keys():
                reasoning_output = event["reasoning_output"]

        if event["messages"]:
            last_message = event["messages"][-1]
            if isinstance(last_message, HumanMessage):
                final_response = last_message.content
    end_time = time.time()  
    time_taken = end_time - start_time
    time_stats[reasoning_n_bot_response_time_profiling_label].append(time_taken)
    logging.info(f"Time taken for Reasoning + final response: {time_taken:.4f} seconds")  
    # -------------------------------------------------------------------------- #

    # ------------------------- Step 3: Refined final response ------------------------- #
    if final_response:
        if reasoning_flag:
            # Save reasoning and response to logs
            logs.append({
                "message_id": message_id,
                "query": query,
                "bot_response": final_response,
                "reasoning": reasoning_output,  
                "agents_prompt": agents_prompt
            })
        else:
            # Save reasoning and response to logs
            logs.append({
                "message_id": message_id,
                "query": query,
                "bot_response": final_response,
                "agents_prompt": agents_prompt
            })
        start_time = time.time()
        #refined_response = await refine_final_response(final_response, response_refining_model)
        end_time = time.time()  
        time_taken = end_time - start_time
        time_stats[refined_response_time_profiling_label].append(time_taken)
        logging.info(f"Time taken for refining final response: {time_taken:.4f} seconds")  
        yield ("detailed", final_response, tools.metadata_list, logs)
    # -------------------------------------------------------------------------- #

    avg_time_taken = await average_time_taken(time_stats)
    await save_to_csv(avg_time_taken, session_id)

    save_logs(logs,session_id)
