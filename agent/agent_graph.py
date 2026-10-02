from collections.abc import Sequence
from typing import Literal
 
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
 
from langgraph.graph import (
    END,
    START,
    MessagesState,
    StateGraph,
)
 
from langgraph.prebuilt import ToolNode
 
 
def build_agent_graph(
    *,
    agent_name: str,
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
):
    """
    Build the actual LangGraph workflow used by an existing AI agent.
 
    Graph:
 
        START
          |
          v
        agent
          |
          +---- tool call ----> tools
          |                      |
          |                      v
          +<---------------------+
          |
          +---- no tool call ---> END
 
    The agent_name is intentionally preserved so the source code clearly
    associates the existing paloalto-agent / paloalto-rag-agent with this
    StateGraph implementation.
    """
 
    if not tools:
        raise ValueError(
            f"{agent_name}: at least one tool is required"
        )
 
    bound_model = model.bind_tools(
        list(tools)
    )
 
    async def call_model(
        state: MessagesState,
    ) -> dict:
        """
        Invoke the LLM that powers this specific AI agent.
        """
 
        messages = [
            SystemMessage(
                content=system_prompt
            ),
            *state["messages"],
        ]
 
        response = await bound_model.ainvoke(
            messages
        )
 
        return {
            "messages": [
                response
            ]
        }
 
    def should_continue(
        state: MessagesState,
    ) -> Literal["tools", "__end__"]:
        """
        Route to ToolNode when the LLM generated a tool call.
        Otherwise finish the agent workflow.
        """
 
        messages = state["messages"]
 
        if not messages:
            return END
 
        last_message = messages[-1]
 
        tool_calls = getattr(
            last_message,
            "tool_calls",
            None,
        )
 
        if tool_calls:
            return "tools"
 
        return END
 
    graph = StateGraph(
        MessagesState
    )
 
    graph.add_node(
        "agent",
        call_model,
    )
 
    graph.add_node(
        "tools",
        ToolNode(
            list(tools)
        ),
    )
 
    graph.add_edge(
        START,
        "agent",
    )
 
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
 
    graph.add_edge(
        "tools",
        "agent",
    )
 
    compiled_graph = (
        graph.compile()
    )
 
    # Helpful metadata for runtime/debugging and static inspection.
    compiled_graph.name = agent_name
 
    return compiled_graph