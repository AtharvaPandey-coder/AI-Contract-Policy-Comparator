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



comparison_prompt = ChatPromptTemplate.from_template("""
You are an expert contract and policy change-analysis assistant.

Your task is to compare one clause from an older policy
with one clause from a revised policy.

DOCUMENT A = ORIGINAL / OLDER VERSION
DOCUMENT B = REVISED / NEWER VERSION

ORIGINAL CLAUSE:
{clause_a}

REVISED CLAUSE:
{clause_b}

FOLLOW THESE RULES CAREFULLY:

1. IDENTIFY THE CHANGE TYPE

- No Change:
  The meaning, rights, obligations, permissions, and conditions
  remain substantially identical.

- Reworked:
  The wording or sentence structure changed, but the practical
  meaning and obligations remain substantially the same.

- Materially Changed:
  A right, obligation, permission, restriction, deadline,
  retention period, consent requirement, security control,
  or other meaningful policy condition has changed.

- Removed:
  A clause existed in the original version but is absent
  from the revised version. Use this only when the supplied
  revised clause explicitly indicates that no corresponding
  clause exists.

- Added:
  A clause exists in the revised version but has no
  corresponding clause in the original version. Use this only
  when the supplied original clause explicitly indicates
  that no corresponding clause exists.

2. CHECK THESE CHANGE CATEGORIES

Carefully compare:
- Personal data collected
- Data storage and retention periods
- Data sharing and consent requirements
- Employee rights, access, correction, and deletion
- Security safeguards and access controls
- Third-party and subcontractor permissions
- Automated decision-making and human review
- Reporting deadlines and notification requirements
- Exceptions, restrictions, and compliance obligations

These are examples, not an exhaustive list.

3. ASSIGN RISK LEVEL

- Low: Administrative or wording changes with little practical impact.
- Medium: A meaningful change with limited or manageable impact.
- High: A change that significantly increases exposure, weakens
  protections, removes important rights or safeguards, or creates
  substantial operational or compliance concerns.

Assess risk using only the supplied clauses. Do not assume a law
has been violated without sufficient information.

4. EXPLANATION

- Explain the specific difference between the two clauses.
- State why the difference matters in practice.
- For numeric changes, mention both the old and new values.
- Do not describe an unchanged condition as a new change.
- Do not invent facts or speculate beyond the supplied text.
- Keep the explanation to 1-2 concise sentences.

Return only the output matching the supplied structured schema.
""")



def comparison_clause(clause_a:str,clause_b:str):
    chain = comparison_prompt | structured_llm
    result=chain.invoke({'clause_a':clause_a,'clause_b':clause_b})
    return result.model_dump()






