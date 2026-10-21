from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from typing import Literal
import os
from dotenv import load_dotenv
load_dotenv()
llm=ChatGroq(model='openai/gpt-oss-120b',temperature=0,api_key=os.getenv('GROQ_API_KEY'))


class Verdict(BaseModel):
    change_type:Literal['No Change','Reworked','Materially Changed','Removed','Added']
    risk_level:Literal['Low','Medium','High']
    explaination:str= Field(description="1-2 sentences on what changed and why it matters")


structured_llm=llm.with_structured_output(Verdict,method='json_schema')

comparison_prompt=ChatPromptTemplate.from_template("""
You are a contract and policy comparison assistant.

Document A is the ORIGINAL, OLDER version.
Document B is the REVISED, NEWER version.

Compare the clauses carefully.

Rules:
- No Change: Meaning and obligations remain substantially the same.
- Reworked: Wording changed, but the meaning remains substantially the same.
- Materially Changed: Rights, obligations, duration, permissions, or risks changed.
- Removed: An original clause is absent from the revised document.
- Added: A clause exists in the revised document but not the original.
- Assess risk based on the practical impact of the change.
- Do not invent facts that are not present in the clauses.
- If a placeholder says no corresponding clause exists, treat it as a missing clause.
- If a clause is added or removed, explain the practical implication.

ORIGINAL CLAUSE:
{clause_a}

REVISED CLAUSE:
{clause_b}
""")


def comparison_clause(clause_a:str,clause_b:str):
    chain = comparison_prompt | structured_llm
    result=chain.invoke({'clause_a':clause_a,'clause_b':clause_b})
    return result.model_dump()






