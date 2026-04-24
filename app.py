from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# ──────────────────────────────────────────────
# Context Break Demonstration (Naïve Invocation)
# Each message is sent in isolation — no history.
# ──────────────────────────────────────────────
def naive_invocation(llm, message):
    print("=" * 60)
    print(f"Naive_Invocation Sending:")
    for m in message:
        print(f"{m.content}")
    print("=" * 60)

    # OpenAI — no prior context
    openai_response = llm.invoke(message)
    print(f"[OpenAI]  {openai_response.content}")
    print(f"[OpenAI]  Response type: {type(openai_response)}\n")


# ──────────────────────────────────────────────
# Context Fix Using Messages
# Full conversation history is passed each time.
# ──────────────────────────────────────────────
def context_fix(llm, conversation_history):
    print("=" * 60)
    print(f"Context_Fix Sending:")
    for m in conversation_history:
        print(f"{m.content}")
    print("=" * 60)

    # OpenAI — full history
    openai_response = llm.invoke(conversation_history)
    print(f"[OpenAI]  {openai_response.content}")
    print(f"[OpenAI]  Response type: {type(openai_response)}\n")
    return AIMessage(content=openai_response.content)


def main():
    llm_openai = ChatOpenAI(model="gpt-4.1-nano")
    prompt_1 = [SystemMessage(content="You are a concise, professional, and friendly assistant. Always respond in a funny manner."),
    HumanMessage(content="What is Agentic AI?")]
    prompt_2 = [HumanMessage(content="What are its applications?")]
    prompts = [prompt_1, prompt_2]

    for prompt in prompts:
        naive_invocation(llm=llm_openai, message=prompt)

    conversation_history = []
    # Accumulates across turns
    for prompt in prompts:
        conversation_history += prompt
        conversation_history.append(context_fix(llm=llm_openai, conversation_history=conversation_history))


if __name__ == "__main__":
    main()

"""
Reflection:
1. Why did string-based invocation fail?
Answer: String-based invocation fails because it doesn't maintain any context between turns, leading to responses that don't consider previous parts of the conversation.
2. Why does message-based invocation work?
Answer: Message-based invocation works because it allows us to maintain a conversation history, enabling the model to understand the context and provide coherent responses that build on previous interactions.
3. What would break in a production AI system if we ignore message history?
Answer: In a production AI system, ignoring message history would lead to disjointed and confusing conversations, as the model would not have the necessary context to provide meaningful responses.
"""