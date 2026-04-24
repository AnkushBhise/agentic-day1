# Agentic AI Day 1: Context Management in LLM Systems

## Overview
This project demonstrates a critical issue in LLM-based applications: **context breaks** when messages are sent in isolation without maintaining conversation history.

The goal is to show:
- ❌ **Naive Approach**: Sending each message independently (loses context)
- ✅ **Correct Approach**: Maintaining full conversation history (preserves context)

## Problem Statement
When building real-world applications with LLMs, simply passing individual messages without maintaining a structured conversation history leads to:
- Lost context from previous interactions
- Inconsistent behavior and responses
- Poor user experience in multi-turn conversations
- Inability for the LLM to make informed decisions based on prior exchanges

## Solution
Using **LangChain's message history** pattern to:
1. Maintain a `SystemMessage` with instructions/role definition
2. Store all `HumanMessage` inputs
3. Retain all `AIMessage` responses
4. Pass the complete history on each invocation

## Project Structure
```
agentic-day1/
├── app.py              # Main demonstration script
├── requirements.txt    # Project dependencies
├── README.md          # This file
└── .env               # Environment variables (API keys)
```

## Technologies Used
- **LangChain**: Framework for LLM orchestration
- **OpenAI API**: GPT-4 model
- **Google Generative AI**: Gemini models (optional)
- **Python 3.8+**: Core language

## Key Dependencies
- `langchain` - LLM framework
- `langchain-openai` - OpenAI integration
- `langchain-core` - Core LLM abstractions
- `langgraph` - Workflow orchestration
- `python-dotenv` - Environment management
- `pytest` - Testing framework

## Setup

### 1. Clone and Navigate
```bash
cd /Users/ankushbhise/AgenticAi_Projects/agentic-day1
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here  # Optional
```

### 5. Run the Application
```bash
python app.py
```

## How It Works

### Naive Invocation
```python
def naive_invocation(llm, message):
    # Each call loses context — no conversation history
    response = llm.invoke(message)
    return response
```

**Problem**: If you ask "What is Agentic AI?" followed by "What are its applications?", the LLM won't understand the second message refers to Agentic AI.

### Context Fix
```python
def context_fix(llm, conversation_history):
    # Full history passed on each invocation
    response = llm.invoke(conversation_history)
    return AIMessage(content=response.content)
```

**Solution**: Pass all previous messages (`SystemMessage`, `HumanMessage`, `AIMessage`) to maintain context.

## Example Output
```
============================================================
Naive_Invocation Sending:
You are a concise, professional, and friendly assistant...
What is Agentic AI?
============================================================
[OpenAI] Agentic AI refers to autonomous systems...

============================================================
Context_Fix Sending:
You are a concise, professional, and friendly assistant...
What is Agentic AI?
...previous response...
What are its applications?
============================================================
[OpenAI] Based on our discussion about Agentic AI, its applications include...
```

## Key Learnings
1. **State Management**: LLMs are stateless; you must provide full context
2. **Message Types**: Use `SystemMessage`, `HumanMessage`, `AIMessage` appropriately
3. **Conversation Patterns**: Always append new messages to history
4. **API Efficiency**: Larger histories = larger API calls (consider truncation strategies)

## Next Steps
- Implement message history persistence (database)
- Add token counting to optimize history length
- Implement sliding window for long conversations
- Test with different LLM providers
- Build a web interface for interactive testing

## License
See [LICENSE](LICENSE) file for details.

## Contributing
This is an educational project. Feel free to fork and experiment!
