from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from typing import Literal
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
RISK_LEVEL:[Low / Medium / High]
EXPLAINATION: <1-2 sentences on what Changed and why it matters >.
""")

def comparison_clause(clause_a:str,clause_b:str)->dict:
    """ Sends one alingned clause pair to the LLm and parses its structured verdict."""
    chain=comparison_prompt | llm
    result=chain.invoke({'clause_a':clause_a,'clause_b':clause_b})
    return parse_response(result.content)


def parse_response(text:str)->dict:
    """Turns the LLm's structure text reply into a dict for the UI table"""
    print(repr(text))
    result ={'change_type':"",'risk_level':"",'explaination':""}
    current=None
    for line in text.strip().split("\n"):
        line=line.replace("*","").strip()
        if line.upper().startswith("CHANGE_TYPE:"):
            result['change_type']=line.split(":",1)[1].strip()
        elif line.upper().startswith("RISK_LEVEL"):
            result['risk_level']=line.split(":",1)[1].strip()
        elif line.upper().startswith("EXPLAIN"):
            result["explaination"]=line.split(":",1)[1].strip()
            current='explaination'
        elif current=='explaination' and line:
            result['explaination'] += " "+ line

    r=result['risk_level'].lower()
    if "high" in r:
        result['risk_level']='High'
    elif "med" in r:
        result['risk_level']='Medium'
    elif 'low' in r:
        result['risk_level']='Low' 
        
    return result







