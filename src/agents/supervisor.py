from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser

def create_supervisor_node(model, supervisor_system_prompt, members):
    """
    Create a supervisor node for managing the workflow.

    Args:
        model: The model to use for the supervisor node.
        supervisor_system_prompt (str): The system prompt for the supervisor.
        members (list): List of team members.

    Returns:
        A supervisor chain object configured with the given model and prompts.
    """
    options = members
    print("\n\t [INFO]: Supervisor Team Members : ", options)

    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", supervisor_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Using a chain-of-thought approach, step by step, analyze the conversation above to determine the next best action."
                "Context Reflection: Begin by reflecting on the context of the conversation and each team member's contributions so far."
                "Consider the progress that has been made and any challenges that have arisen."
                "Team Assessment: Evaluate how each team member can effectively contribute to moving the task forward."
                "Identify any gaps or opportunities where a specific member's skills or input would be most valuable."
                "Decision Making: Based on your analysis, decide who should act next to ensure the task progresses smoothly."
                "Choose one of the following options: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    supervisor_chain = (
        prompt
        | model.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    return supervisor_chain
