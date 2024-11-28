import sys
from src.utils import (
    load_config, 
    get_active_client_info,
    create_client_manifest_path,
    initialize_models,
    initialize_vectorstore,
    initialize_retriever,
    setup_database,
    initialize_sql_agent
)
# --------- Main to initialize and setup all components based on the configuration --------- #

conversation_logs_dir = './conversation_logs'
settings_filepath = './src/settings.yaml'
product_information_filepath = './src/product_information.yaml'
intent_file_path= './src/intent.yaml'
config = load_config(settings_filepath)
supervisor_agent_name  = config['paths']['supervisor_agent_name']
client_information = get_active_client_info(config)

if not client_information:
    print("[INFO]: No active client found.")
    sys.exit(1)

client_name = client_information['name']
print("[INFO]:\nCLIENT : ", client_information,"\n")

tool_manifests_file_path  = create_client_manifest_path(client_name, config['paths']['tool_manifests_file_path'])
agent_manifests_file_path = create_client_manifest_path(client_name, config['paths']['agent_manifests_file_path'])

(   model, 
    embeddings_model, 
    memory, 
    quick_response_model, 
    response_refining_model, 
    chatbot_responses_model_map
) = initialize_models(config, client_name)

vectorstore = initialize_vectorstore(config, embeddings_model)
retriever_self_query = initialize_retriever(model, vectorstore, config)

tools_package = f"src.tools.Manifests.{client_name}.tools"

if 'database' in client_information:
    print("[INFO]: Database information provided in client information.")
    db_info = client_information['database']
    host = db_info['host']
    username = db_info['username']
    password = db_info['password']
    port = db_info['port']
    database = db_info['name']

    sql_db = setup_database(host, username, password, port, database)
    sqldb_agent = initialize_sql_agent(model, sql_db)
else:
    print("[INFO]: Database information not provided in client information.")
