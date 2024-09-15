from rag.functions.investor_function import functions_map
import json

def get_run(run_id, thread_id, client):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def send_message_to_thread(thread_id, content, client):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )

def get_messages(thread_id, client):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    final_message = ""
    for message in messages:
        if message.role == " assistant":
            final_message = message.content[0].text.value
    return final_message


def get_tool_outputs(run_id, thread_id, client):
    run = get_run(run_id, thread_id, client)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id, client):
    output = get_tool_outputs(run_id, thread_id, client)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id, thread_id=thread_id, tool_outputs=output
    )