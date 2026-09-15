"""Runs one Galaxy MCP tool-calling session, then exits.

This lives in a subprocess because the Gemini and local-model clients start
their own event loop, which clashes with the server's. It reads a JSON job on
stdin and writes a JSON result on stdout, so nothing -- token, API key or user
query -- is ever interpolated into source or written to disk.

    echo '{"provider": "gemini", ...}' | python -m app.Galaxy_integration.mcp_worker
"""
import asyncio
import json
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

MAX_TOOL_ROUNDS = 10


def _tool_schema(tool):
    """JSON schema for a LangChain tool, across pydantic v1/v2 and tools with none."""
    for accessor in ("model_json_schema", "schema"):
        try:
            return getattr(tool.args_schema, accessor)()
        except Exception:
            continue
    return {"type": "object", "properties": {}}


async def _load_tools(mcp_url, token):
    client = MultiServerMCPClient({
        "galaxyTools": {
            "transport": "streamable_http",
            "url": mcp_url,
            "headers": {"Authorization": f"Bearer {token}"},
        }
    })
    return await client.get_tools()


async def _run_gemini(job, tools):
    import google.generativeai as genai

    genai.configure(api_key=job["api_key"])
    declarations = [
        {"name": t.name, "description": t.description or "", "parameters": _tool_schema(t)}
        for t in tools
    ]
    model = genai.GenerativeModel(
        job.get("model") or "gemini-2.5-flash",
        tools=[{"function_declarations": declarations}],
    )
    chat = model.start_chat()
    response = chat.send_message(job["query"])

    for _ in range(MAX_TOOL_ROUNDS):
        calls = [
            part
            for candidate in response.candidates
            for part in candidate.content.parts
            if getattr(part, "function_call", None) and part.function_call.name
        ]
        if not calls:
            return response.text

        replies = []
        for part in calls:
            call = part.function_call
            tool = next((t for t in tools if t.name == call.name), None)
            if tool is None:
                continue
            result = await tool.ainvoke(dict(call.args))
            replies.append({
                "function_response": {"name": call.name, "response": {"result": str(result)}}
            })
        response = chat.send_message(replies)

    return response.text


async def _run_openai_compatible(job, tools):
    import openai

    client = openai.OpenAI(base_url=f"{job['host']}/v1", api_key=job["api_key"])
    specs = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": _tool_schema(t),
            },
        }
        for t in tools
    ]
    messages = [{"role": "user", "content": job["query"]}]

    for _ in range(MAX_TOOL_ROUNDS):
        completion = client.chat.completions.create(
            model=job["model"], messages=messages, tools=specs or None
        )
        choice = completion.choices[0]
        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            return choice.message.content or ""

        messages.append(choice.message)
        for call in choice.message.tool_calls:
            tool = next((t for t in tools if t.name == call.function.name), None)
            if tool is None:
                continue
            result = await tool.ainvoke(json.loads(call.function.arguments))
            messages.append({
                "role": "tool", "tool_call_id": call.id, "content": str(result)
            })

    return ""


RUNNERS = {"gemini": _run_gemini, "local_model": _run_openai_compatible}


async def _main():
    job = json.loads(sys.stdin.read())
    runner = RUNNERS.get(job["provider"])
    if runner is None:
        raise ValueError(f"unsupported provider: {job['provider']}")
    tools = await _load_tools(job["mcp_url"], job["token"])
    return await runner(job, tools)


if __name__ == "__main__":
    try:
        print(json.dumps({"text": asyncio.run(_main())}))
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)
