from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
llm=ChatGroq(model='openai/gpt-oss-120b',temperature=0,api_key=os.getenv('GROQ_API_KEY'))
comparison_prompt=ChatPromptTemplate.from_template("""
You are a contract review assistant.Compare these two clause versions.


CLAUSE FROM DOCUMENT A
{clause_a}
CLAUSE FROM DOCUMENT B
{clause_b}

Respond in this exact format:
CHANGE_TYPE: [No Change / Reworked (same meaning) / Materially Changed / Removed / Added]
RISK_LEVEL:[Low / Medium / Hard]
EXPLAINATION: <1-2 sentences on what Changed and why it matters >.
""")

def comparison_clause(clause_a:str,clause_b:str)->dict:
    """ Sends one alingned clause pair to the LLm and parses its structured verdict."""
    chain=comparison_prompt | llm
    result=chain.invoke({'clause_a':clause_a,'clause_b':clause_b})
    return parse_response(result.content)


def parse_response(text:str)->dict:
    """Turns the LLm's structure text reply into a dict for the UI table"""
    result ={'change_type':"",'risk_level':"",'explaination':""}
    for line in text.strip().split("\n"):
        if line.startswith("CHANGE_TYPE:"):
            result['change_type']=line.replace("CHANGE_TYPE:","").strip()
        elif line.startswith("RISK_LEVEL"):
            result['risk_level']=line.replace("RISK_LEVEL:","").strip()
        elif line.startswith("EXPLAINATION"):
            result['explaination']=line.replace("EXPLAINATION","").strip()
    return result










