import yaml
import redis
import hashlib
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from src.config import sqldb_agent, model,  retriever_self_query, model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from src.config import product_information_filepath, model
from src.utils import get_product_information, inject_product_info_into_tool

product_information = get_product_information(product_information_filepath)

print("\n\nUSING PRODUCT INFORMATION for Tools: --------",product_information)

metadata_list=[]

tools_prompt_templates_dict = {
    "subjective_terms_handling": """
You will receive a user query that contains subjective terms. Your task is to:
1. Identify Subjective term in the user’s query that imply a preference or quality judgment. These might include terms related to quality, value, or ranking such as 'premium', 'best', 'top', 'excellent', etc.
2. Use your LLM knowledge to choose the most relevant criteria only one for the subjective term. 
#IMPORTANT#- ALWAYS respond in a way: First, acknowledge the user. Then, for the given preference, use your LLM knowledge to select the most relevant [criteria] (e.g., popular choice, reputation, etc.). Suggest the best options based on this [selected criteria], and briefly explain why they are ideal. Additionally, mention one or two other relevant criteria to consider (e.g., vintage year, customer reviews). Ask the user if they would like to consider these other criteria.
""",

    "recommend_product_not_found": """
You are a {product} recommendation assistant. The user asked: '{user_query}'. Only reply in this way: Unfortunately, there are no matching {product} found in the database for this request. Provide a polite and apologetic response to the user, expressing regret for not having the requested {product}s available."
""",

    "recommend_product": """
You are a {product} recommendation assistant. Based on the following retrieved documents, provide a detailed recommendation for each {product}. You must categorize each {product} using at least one of the following tags if they have. Ensure that every {product} is assigned a tag, and use reasoning to determine how each {product} stands out. Here are the details of the retrieved {product}:\n
{retrieved_data}\n
Use the following criteria to categorize each {product}:\n
- **Closest Match**: Tag a {product} as 'Closest Match' if it best fits the user's specific criteria.\n
- **Best Value**: Tag a {product} as 'Best Value' if it offers exceptional value for the price. This could be a {product} that provides high quality at a relatively low price or a {product} that has a special offer or discount. Highlight {product}s that offer a great deal for their price or those that are currently available at a reduced price.\n
- **Good Value**: Tag a {product} as 'Good Value' if it is a budget-friendly option but may not be the best deal or offer the highest quality in the list. This indicates that the {product} is reasonably priced and provides a satisfactory experience for its cost, though it might not be the cheapest or most highly discounted option available.\n
- **On Sale**: Tag a {product} as 'On Sale' if it is currently available at a reduced price compared to its usual cost. This includes {product}s that have been discounted or are part of a special promotion. Emphasize the discount or special offer to highlight how much the user can save by purchasing it.\n
Make sure to apply at least one of the tags to each {product} in {retrieved_data}. Provide relevant reasoning based on the criteria mentioned. For example, if a {product} meets the user's preferences perfectly and is also on sale, tag it as both 'Closest Match' and 'On Sale'.
""",

    "sql_tool": """
Use this query to convert it into sql query, DO NOT apply the term wine in SQL queries for liquor_style types. Use the term exactly as selected by the user. DO not apply wine with red , white and sparkling, use them as it is red, white, sparkling ...etc
"""
}



for tool_name, tool_prompt in tools_prompt_templates_dict.items():
    tool_info = {'name':tool_name, 'tool_data': tool_prompt}

    print("\n\n\n\t\ttool_name : ", tool_name)
    print("\n----------------------------------------> OLD PROMPT : ", tool_prompt)

    tools_prompt_templates_dict[tool_name] = inject_product_info_into_tool(
        model=model, 
        tool_info=tool_info,
        product_information=product_information,
        use_case='tool_prompt'
    )
    print("\n----------------------------------------> NEW PROMPT : ", tools_prompt_templates_dict[tool_name])



@tool
def process_image(image_data: str):
    """this tool expects binary image, processes the image, analyzes appearance, and returns a greeting.
       Greeting should be dynamic based on clothing, age, glasess, gender and could be anything else.
    """
    print(image_data[0:100])
    return ["greeting"]
    
@tool()
def greet():
    """{greet}"""
    return "greet"

@tool()
def ask_entities():

    """{ask_entities}"""
    return ask_entities

@tool()
def subjective_terms_handling(query: str):
    """{subjective_terms_handling}"""

    # prompt = """
    # You will receive a user query that contains subjective terms. Your task is to:
    #    1. Identify Subjective term in the user’s query that imply a preference or quality judgment. These might include terms related to quality, value, or ranking such as 'premium', 'best', 'top', 'excellent', etc.
    #    2. Use your LLM knowledge to choose the most relevant criteria only one for the subjective term. 
    #    #IMPORTANT#- ALWAYS respond in a way: First, acknowledge the user. Then, for the given preference, use your LLM knowledge to select the most relevant [criteria] (e.g., popular choice, reputation, etc.). Suggest the best options based on this [selected criteria], and briefly explain why they are ideal. Additionally, mention one or two other relevant criteria to consider (e.g., vintage year, customer reviews). Ask the user if they would like to consider these other criteria.
    #    """

    prompt = tools_prompt_templates_dict['subjective_terms_handling']
    response = model.predict(prompt.format(user_query=query))

    print('RESPONSE---------------------------------------------------------', response, "------------------------------------------------------------")
    return response

@tool
def recommend_product(query: str):
    """{recommend_product}"""
    
    global metadata_list
    
    # Preprocess the query
    query = "suggest me a " + query.lower()
    print("query--------------------------------", query)
    
    # Fetch documents using the query
    fetching_output = retriever_self_query.get_relevant_documents(query)
    print("--------------------------------------------")
    print(f"Number of documents retrieved: {len(fetching_output)}")
    print("--------------------------------------------")
    
    metadata_list = [doc.metadata for doc in fetching_output]

    if len(fetching_output) == 0:
        # prompt_template = (
        #     "You are a wine recommendation assistant. The user asked: '{user_query}'. Only reply in this way: Unfortunately, there are no matching wines found in the database for this request. Provide a polite and apologetic response to the user, expressing regret for not having the requested wines available."
        # )
        prompt_template = tools_prompt_templates_dict['recommend_product_not_found']
        prompt = prompt_template.format(user_query=query)
        message = model.predict(prompt)
        return message

    page_content = [doc.page_content for doc in fetching_output]

    product_details = []
    for content, metadata in zip(page_content, metadata_list):
        details = {"name": metadata.get('name'),"price": metadata.get('price'),"offer": metadata.get('offer'), "page_content": content}
        product_details.append(details)

    # Format product details into a readable format for the prompt
    formatted_product_details = "\n".join(
        [f"Name: {product['name']}\nPrice: {product['price']}\nOffer: {product['offer']}\nDetails: {product['page_content']}" for product in product_details]
    )

#     prompt_template = (
#     "You are a wine recommendation assistant. Based on the following retrieved documents, provide a detailed recommendation for each wine. You must categorize each wine using at least one of the following tags if they have. Ensure that every wine is assigned a tag, and use reasoning to determine how each wine stands out. Here are the details of the retrieved wines:\n"
#     "{product_details}\n"
#     "Use the following criteria to categorize each wine:\n"
#     "- **Closest Match**: Tag a wine as 'Closest Match' if it best fits the user's specific criteria.\n"
#     "- **Best Value**: Tag a wine as 'Best Value' if it offers exceptional value for the price. This could be a wine that provides high quality at a relatively low price or a wine that has a special offer or discount. Highlight wines that offer a great deal for their price or those that are currently available at a reduced price.\n"
#     "- **Good Value**: Tag a wine as 'Good Value' if it is a budget-friendly option but may not be the best deal or offer the highest quality in the list. This indicates that the wine is reasonably priced and provides a satisfactory experience for its cost, though it might not be the cheapest or most highly discounted option available.\n"
#     "- **On Sale**: Tag a wine as 'On Sale' if it is currently available at a reduced price compared to its usual cost. This includes wines that have been discounted or are part of a special promotion. Emphasize the discount or special offer to highlight how much the user can save by purchasing it.\n"
#     "Make sure to apply at least one of the tags to each wine in {product_details}. Provide relevant reasoning based on the criteria mentioned. For example, if a wine meets the user's preferences perfectly and is also on sale, tag it as both 'Closest Match' and 'On Sale'."
# )

    prompt = tools_prompt_templates_dict['recommend_product']
    prompt = prompt_template.format(retrieved_data=formatted_product_details)
    output = model.predict(prompt)
    return output



redis_client = redis.Redis(host='localhost', port=6379)


def get_cache_key(query: str):
   
    return hashlib.md5(query.encode()).hexdigest()


@tool()
def sql_tool(query: str):
    """
    {sql_tool}
    """
    cache_key = get_cache_key(query)
    print('cache_key-------------', cache_key)
    

    cached_result = redis_client.get(cache_key)
    if cached_result:
        print('cached_result----------', cached_result)
        print("[INFO]: Returning cached result from Redis.")
        return cached_result.decode('utf-8')

   
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", tools_prompt_templates_dict['sql_tool']),
            ("user", "{question}\n ai: "),
        ]
    )
    response = sqldb_agent.run(prompt.format(question=query))
    print('response---------------', response)

    redis_client.set(cache_key, response)
    print("[INFO]: Query executed and result cached in Redis.")

    return response



@tool()
def sql_tool_1(query: str):
    """
    {sql_tool}
    """
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", """create an SQL query based on tsql_tools = [ask_entities, sql_tool]
# he user's question.
#                           - SELECT EXISTS (SELECT 1 FROM wine_reviews WHERE liquor_style = '<liquor_style_value>')
#                           - SELECT EXISTS (SELECT 1 FROM wine_reviews WHERE variety = '<variety_value>')"""),
#             ("user", "{question}\n ai: "),
#         ]
#     )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", tools_prompt_templates_dict['sql_tool']),
            ("user", "{question}\n ai: "),
        ]
    )
    response = sqldb_agent.run(prompt.format(question=query))

    return response

@tool()
def knowledge_base_search(query: str):
    """
    {knowledge_base_search}
    """
    search = TavilySearchAPIWrapper()

    knowledge_base_search = TavilySearchResults(api_wrapper=search, include_link=False)
    results = knowledge_base_search.run(query)
    return results
