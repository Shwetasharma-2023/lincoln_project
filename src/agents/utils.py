import os
import yaml
import logging
import functools
from jinja2 import Template

from src.agents.agents import Agent, agent_node

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def create_member_agent_node(model, name, agent_system_prompt, agent_tools, checkpointer, reasoning_flag):
    """
    Create a node for a member agent.

    Args:
        model: The model to use for the agent.
        name (str): The name of the agent.
        agent_system_prompt (str): The system prompt for the agent.
        agent_tools (list): List of tools available to the agent.
        checkpointer: The checkpointer for state management.

    Returns:
        A partial function that represents the agent node.
    """
    agent = Agent(model=model, tools=agent_tools, checkpointer=checkpointer, system=agent_system_prompt, reasoning_flag=reasoning_flag)
    node = functools.partial(agent_node, agent=agent.graph, name=name, reasoning_flag=reasoning_flag)
    return node

def get_agents_status(yaml_file):
    """
    Get the status of agents from a YAML file.

    Args:
        yaml_file (str): Path to the YAML file.

    Returns:
        tuple: A dictionary of agents' statuses and the background information filename.
    """
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    agents_status_dict = {agent['name']: agent['status'] for agent in data['agents']}
    background_information_filename = data['background_information'][0]['filename']
    return agents_status_dict, background_information_filename

def inject_product_information(file_path, product_information):
    """
    Inject the product information into prompt YAML file content.

    Args:
        file_path (str): Path to the YAML file.
        product_information (dict): Information dictionary of the product.

    Returns:
        str: The modified YAML content with injected product information.
    """
    logging.info(f"Injecting Product Information into {file_path}")

    # Read the YAML file content
    with open(file_path, 'r') as file:
        yaml_file_data = file.read()

    # Extract attribute values
    attributes_list = list(product_information['attributes'].values())

    # Prepare the product context for templating
    product_context = {
        'product': product_information['product'],
        'all_attributes_name': product_information['all_attributes_name'],
        'no_of_attributes': product_information['no_of_attributes']

    }
    for idx, attribute in enumerate(attributes_list, start=1):
        product_context[f'attribute{idx}'] = attribute

    print(75 * "_")
    print("--------Using Product Context for Agent--------")
    print(product_context)
    print(75 * "_")

    # Create a Jinja2 Template and render it with the product context
    template = Template(yaml_file_data)
    modified_yaml_content = template.render(product_context)

    return modified_yaml_content

def read_background_information(yaml_file):
    """
    Read background information from a YAML file.

    Args:
        yaml_file (str): Path to the YAML file.

    Returns:
        str: The system prompt from the background information.
    """
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    system_prompt = data['intents']['system_prompt']
    return system_prompt

def generate_prompt(yaml_file_content, intent_name, members=[]):
    """
    Generate a prompt for an agent based on the background information and intent.

        Args:
        yaml_file_content (str): yaml content with injected product information.
        intent_name (str): The name of the intent.
        product_information (dict): The information of product to sell.
        members (list): List of member agents.

    Returns:
        tuple: The dynamic system prompt and entities prompt.
    """

    # Load the YAML data with Product information injected
    data = yaml.safe_load(yaml_file_content)

    # Extract data under the specified intent
    intent_data = data['intents'][intent_name]
    entities = intent_data['entities']
    system_prompt = intent_data['system_prompt']
    entities_prompt = None
    magic_prompt = system_prompt['magic_prompt']
    role = system_prompt['role']
    goal = system_prompt['goal']

    if intent_name == 'supervisor':
        role += f"{members}"
    task = system_prompt['backstory']

    if entities:
        task += f" Collect information on the following entities: {', '.join(entities)}."
        entities_prompt = f"""
        Your task is to interact with the user and gather all the necessary entities: {', '.join(entities)}. Be conversational and context-aware, seamlessly integrating the questions into the flow of the conversation. Only ask for entities that have not yet been provided, ensuring not to repeat any queries. Approach the collection of each entity naturally, based on the current state of the conversation. The goal is to gather the required information without making the interaction feel like a checklist.
        """
    
    dos = system_prompt['instructions']['dos']
    donts = system_prompt['instructions']['donts']

    examples = system_prompt['examples']
    penalty = system_prompt['penalty']

    # Combine examples into one variable
    #examples = "\n".join([f"  {key}: {value}" for example in system_prompt['examples'] for key, value in example.items()])

    # Create the dynamic system prompt
    dynamic_system_prompt = f"""
      {magic_prompt} \n 
      Role: {role} \n
      Goal: {goal} \n
      Task: {task} \n
      Instructions: \n
      {dos} \n 
      {donts} \n 
      Example Responses: \n
      {examples} \n 
      Panelty Warning:\n 
      {penalty}
    """
    print("-" * 100)
    print(f"{intent_name} : {dynamic_system_prompt}")
    print("-" * 100)
    return dynamic_system_prompt, entities_prompt



def inject_intent_and_product_data_into_yaml(file_path, intent_data, product_information):
    """
    Inject intent data and product context into the Supervisor agent's YAML file.

    Args:
        file_path (str): Path to the YAML file of the Supervisor agent.
        intent_data (dict): Intent data to inject into the YAML file.
        product_information (dict): Product information to inject into the YAML file.

    Returns:
        str: The modified YAML content with injected intent and product data.
    """
    with open(file_path, 'r') as file:
        yaml_file_data = file.read()

    # Extract and prepare intent context
    intent_names = []
    objectives = []
    all_steps = []
    for intent in intent_data['intent']:
        intent_names.append(intent.get('name'))
        objectives.extend(intent.get('objectives', []))
        all_steps.extend(intent.get('sales_steps', []))
    
    intent_context = {
        'intent_name': intent_names,
        'objectives': objectives,
        'sales_steps': all_steps
    }

    # Extract attribute values
    attributes_list = list(product_information['attributes'].values())
    product_context = {
        'product': product_information['product'],
        'all_attributes_name': product_information['all_attributes_name'],
        'no_of_attributes': product_information['no_of_attributes']
    }
    for idx, attribute in enumerate(attributes_list, start=1):
        product_context[f'attribute{idx}'] = attribute

    # Merge intent context and product context
    merged_context = {**intent_context, **product_context}

    # Render the YAML file with the merged context
    template = Template(yaml_file_data)
    modified_yaml_content = template.render(merged_context)

    # Debugging output
    print("intent_context----------", intent_context)
    print("product_context---------", product_context)
    print("merged_context----------", merged_context)
    print("modified_yaml_content----------------------------------", modified_yaml_content)

    return modified_yaml_content


def generate_active_agents_prompt(agent_manifests_file_path, product_information, intent_data):
    """
    Generates prompts for active agents as specified in the manifest file.
    
    Args:
        agent_manifests_file_path (str): Path to the agent manifests file.
        product_information (dict): information of product
    
    Returns:
        dict: A dictionary where each key is a serial number and each value is another dictionary containing:
            - 'agent_name': str
            - 'system_prompt': str
            - 'tools_entity_prompt': str
    """
    agents_status, background_information_filename = get_agents_status(agent_manifests_file_path)
    
    manifests_dir = os.path.dirname(agent_manifests_file_path)
    background_information = read_background_information(os.path.join(manifests_dir, background_information_filename))
    
    generated_prompts = {}
    agent_count = 1
    for agent_name, status in agents_status.items():
        if status == 'active':
            file_path = os.path.join(manifests_dir, f"{agent_name}.yaml")
            prompt_yaml_with_product_info = inject_intent_and_product_data_into_yaml(file_path, intent_data, product_information)
            system_prompt, tools_entity_prompt = generate_prompt(
                yaml_file_content=prompt_yaml_with_product_info,
                intent_name=agent_name,
            )
            generated_prompts[agent_count] = {
                'agent_name': agent_name,
                'agent_system_prompt': system_prompt,
                'agent_entity_prompt': tools_entity_prompt
            }
            agent_count += 1

    return generated_prompts, background_information
