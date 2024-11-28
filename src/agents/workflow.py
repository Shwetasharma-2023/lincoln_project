import operator
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END, START

async def create_workflow(nodes_dict, supervisor_node_name, checkpointer, reasoning_flag):
    """
    Create a workflow graph based on the provided nodes and supervisor.

    Args:
        nodes_dict (dict): Dictionary of nodes where keys are node names and values are node functions.
        supervisor_node_name (str): The name of the supervisor node.
        checkpointer: The checkpointer for state management.

    Returns:
        StateGraph: The compiled state graph for the workflow.
    """
    if reasoning_flag:
        class SupervisorState(TypedDict):
            """
            A dictionary type for the state of the supervisor, containing messages and the next state (step).
            """
            messages: Annotated[Sequence[BaseMessage], operator.add]
            next: str
            reasoning_output : str
    else:
        class SupervisorState(TypedDict):
            """
            A dictionary type for the state of the supervisor, containing messages and the next state (step).
            """
            messages: Annotated[Sequence[BaseMessage], operator.add]
            next: str

    graph = StateGraph(SupervisorState)

    
    # Add nodes to the graph
    for agent_node_name, agent_node in nodes_dict.items():

        graph.add_node(agent_node_name, agent_node)

    # Create conditional edges for the supervisor
    members = [name for name in nodes_dict.keys() if name != supervisor_node_name]
    conditional_map = {member: member for member in members}
    conditional_map["FINISH"] = END
    graph.add_conditional_edges(
        supervisor_node_name,
        lambda state: state["next"],
        conditional_map
    )
    # Add the starting edge
    graph.add_edge(START, supervisor_node_name)

    # Compile and return the graph
    compiled_graph = graph.compile(checkpointer=checkpointer)
    return compiled_graph
