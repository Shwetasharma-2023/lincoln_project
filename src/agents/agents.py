import json
import logging
import operator
import asyncio
from typing import TypedDict, Annotated
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Reasoning(TypedDict):
    """Structured response combining reasoning and final output."""
    reasoning: Annotated[str, ..., "Provide your reasoning or explanation for why you are making this recommendation or decision."]
    output: Annotated[str, ..., "The final recommendation or answer without any additional explanation."]

class BaseAgentState(TypedDict):
    """Base Class for Agent State"""
    messages: Annotated[list[AnyMessage], operator.add]

class AgentState(BaseAgentState):
    pass

structured_response_prompt ="""   
        **ALWAYS RESPOND in the following JSON format**:
        {"reasoning": brief explanation of why this output is chosen, including context and logic behind the decision,
        "output": actual response, relevant to the user's query or needs}
        Ensure that each response will be in json format with "reasoning" and "output" as a key explains the reasoning first before presenting the actual output. The reasoning should clearly justify why the specific response is provided, followed by the output that answers the user's query directly.
"""

class Agent:
    def __init__(self, model, tools, checkpointer, reasoning_flag, system=""):
        """
        Initialize the Agent with the model, tools, checkpointer, and optional system message.

        Args:
            model: The model to use for the agent.
            tools: The tools available to the agent.
            checkpointer: The checkpointer for state management.
            system (str): Optional system message.
            reasoning_flag (bool): Whether to include reasoning in response
        """
        self.system = system

        if reasoning_flag:
            self.system = system + structured_response_prompt

            class AgentState(BaseAgentState):
                reasoning_output: str
        else: 
            class AgentState(TypedDict):
                messages: Annotated[list[AnyMessage], operator.add]

        self.model = model
        self.tools = {t.name: t for t in tools} if tools else {}
        
        if tools:
            self.model = self.model.bind_tools(tools)
        
        # Initialize the state graph
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)

        graph.add_edge("action", "llm")

        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )

        graph.set_entry_point("llm")
        self.graph = graph.compile() #(checkpointer=checkpointer)

    def exists_action(self, state: AgentState) -> bool:
        """
        Check if the last message in the state has tool calls.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        result = state['messages'][-1]

        return len(result.tool_calls) > 0

    async def call_openai(self, state: AgentState) -> dict:
        """
        Invoke the model with the current state messages.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            dict: The updated state with the model's response.
        """

        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        message = await self.model.ainvoke(messages)

        return {'messages': [message]}

    async def take_action(self, state: AgentState) -> dict:
        """
        Execute tool calls based on the last message in the state.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            dict: The updated state with the tool call results.
        """
        tool_calls = state['messages'][-1].tool_calls
        results = []

        for t in tool_calls:
            logging.info(f"[INFO]: Calling: {t}")

            if t['name'] not in self.tools:  # check for bad tool name from LLM
                logging.info("[INFO]: ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad

            else:
                result = await self.tools[t['name']].ainvoke(t['args'])

            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

        logging.info("[INFO]: Back to the model!")
        return {'messages': results}

async def agent_node(state, agent, name, reasoning_flag=False) -> dict:
    """
        Process the state with the agent and return the result.

        Args:
            state: The current state to process.
            agent: The agent to process the state.
            name: The name of the agent.

        Returns:
            dict: The result of the agent's processing.
        """
    result = await agent.ainvoke(state)
    final_message = result["messages"][-1].content
    print('final_message------------------', final_message)
    if reasoning_flag:
        if final_message.startswith('{') and final_message.endswith('}'):
            final_message = json.loads(final_message)

        is_valid = validate_json(final_message)
        if not is_valid and (reasoning_flag == True):
            safe_message = json.dumps(final_message)
            fix_prompt = f"""Always structure your response as follows:
    
                    {{
                        "reasoning": "Provide a detailed explanation of why you gave this response. Include factors you considered, any assumptions you made, and how you arrived at your conclusion.",
                        "output": "{safe_message}"
                    }}
    
                    Currently, this is the output you provided: "{safe_message}".
                    Do not modify the output, just provide reasoning for it and return in JSON format.
                    """

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
            corrected_response = llm.predict(fix_prompt)
            print("corrected_response-----------------------", corrected_response)

            corrected_response = json.loads(corrected_response)
            reasoning = corrected_response.get('reasoning')
            output = corrected_response.get('output')
            logging.info(f"Corrected reasoning: {reasoning}")
            logging.info(f"Corrected output: {output}")
            if reasoning_flag:
                return {"messages": [HumanMessage(content=output, name=name)],
                        "reasoning_output": reasoning}
            else:
                return {"messages": [HumanMessage(content=output, name=name)]}

        else:
            print(type(final_message))
            reasoning = final_message["reasoning"]
            output = final_message["output"]
            logging.info(f"\n--------------final_message : {final_message} \n\n--------------reasoning : {reasoning} \n\n--------------output : {output}\n")

            if reasoning_flag:
                return {"messages": [HumanMessage(content=output, name=name)],
                        "reasoning_output": reasoning}
            else:
                return {"messages": [HumanMessage(content=output, name=name)]}
    else:
        print("-------------------return-------------------------_")
        return {"messages": [HumanMessage(content=final_message, name=name)]}



