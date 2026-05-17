from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def invoke_llm(prompt: str, model: str = "gpt-oss", temperature: float = 0.0) -> str:
    """
    Centralized utility to handle LLM calls with standardized error handling.
    """
    try:
        llm = ChatOllama(model=model, temperature=temperature)
        # llm = ChatOpenAI(model=model, temperature=temperature)
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"[LLM Error]: {e}")
        # Return a clean fallback message instead of crashing the notebook loop
        return (
            "LLM service unavailable (API timeout, quota exceeded, or billing issue).\n"
            "Please check your local Ollama connection or platform configurations."
        )

def generate_final_response(result) -> str:
    """
    Generates a brief summary in human-readable sentences based on tool results.
    """
    print("generate_final_response input: ", result)

    prompt = f"""
    Generate a brief summary in human readable sentences based on the tool results:
    Result: {result}
    
    Make sure the output are presentable for a streamlit UI.
    """
    return invoke_llm(prompt)
