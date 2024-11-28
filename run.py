import uuid
import asyncio
from langchain_core.messages import SystemMessage
from src.main import initialize_workflow, handle_user_input
from src.utils import get_product_information, extract_intents_objectives_sales_steps
from src.config import product_information_filepath
from src.config import intent_file_path

# ----------------- Product Information ----------------- #
product_information = get_product_information(product_information_filepath)

print('********************************************************')
print("product_information : \n",product_information)
print('********************************************************')
# ------------------------------------------------------------------------------------ #

intent_data= extract_intents_objectives_sales_steps(intent_file_path)
print('********************************************************')
print("intent_data : \n",intent_data)
print('********************************************************')




async def main():
    reasoning_flag = False
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    workflow, biscuit_prompt = await initialize_workflow(reasoning_flag, product_information, intent_data)

    logs = []

    while True:

        async_response_generator= handle_user_input(
            query="",  
            graph=workflow,
            config=config,
            logs=logs,
            reasoning_flag=reasoning_flag,
            session_id = thread_id,
            frontend=False
        )

        async for response in async_response_generator:
            if response[0] == "stop":
                print("\n\n\tStopping")
                return

            if response[0] == "quick":
                print("Quick Response: ", response[1])
            elif response[0] == "detailed":
                 
                final_response = response[1]
                metadata = response[2]
                print("Detailed Response: ", final_response)
                print("Metadata: ", metadata)
                logs = response[3]  

if __name__ == '__main__':
    asyncio.run(main())


