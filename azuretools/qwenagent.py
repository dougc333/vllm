import os
from openai import OpenAI

# Your Azure skills (they load .env themselves)
from azureskills import get_azure_resources, get_azure_billing_info

# ---------- Pick ONE Qwen endpoint ----------

# Option A: Official Qwen API (Alibaba Model Studio / DashScope)
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
)
MODEL = "qwen-plus"   # or whatever model your workspace lists (qwen-max, qwen-turbo...)


# Option B: Local Qwen on your Mac via Ollama (see section 4)
# client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# MODEL = "qwen2.5:7b-instruct"

# ---------- Tool definitions ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_azure_resources",
            "description": "List the Azure resource groups in my subscription and their locations.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_azure_billing_info",
            "description": "Get my total Azure cost so far this month in USD.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

available_functions = {
    "get_azure_resources": get_azure_resources,
    "get_azure_billing_info": get_azure_billing_info,
}

# ---------- Agent loop ----------
def run_agent(user_prompt: str, max_rounds: int = 5) -> str:
    messages = [
        {"role": "system", "content":
         "You are an Azure assistant. Use the provided tools to answer questions "
         "about the user's Azure subscription. Never guess numbers; always call a tool."},
        {"role": "user", "content": user_prompt},
    ]

    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools
        )
        msg = resp.choices[0].message

        # Re-append the assistant message (provider-safe format)
        assistant_msg = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # No tool call -> final answer
        if not msg.tool_calls:
            return msg.content or ""

        # Run each requested skill locally and feed results back
        for tc in msg.tool_calls:
            fn = available_functions.get(tc.function.name)
            result = fn() if fn else "Unknown tool"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": str(result),
            })

    return "Agent stopped: too many tool rounds."


if __name__ == "__main__":
    print(run_agent(
        "What resource groups do I have in Azure, and how much have I spent this month?"
    ))