from langchain.agents import create_agent
from langchain.tools import tool
from document_loader import load_chunk
from comparator import comparison_clause
from clause_matcher import build_index,align_clause
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
llm=ChatGroq(model='openai/gpt-oss-120b',api_key=os.getenv('GROQ_API_KEY'))
@tool
def compare_documents(path_a:str,path_b:str)->str:
    """Compare two contract\ policy PDFs clause by clause and returns a risk-flagged summary"""
    chunks_a=load_chunk(path_a)
    chunks_b=load_chunk(path_b)
    index_b=build_index(chunks_b)

    pairs=align_clause(chunks_a,index_b)
    report_lines=[]
    for pair in pairs:
        verdict=comparison_clause(pair['clause_a'],pair['clause_b'])
        if "no change" not in verdict['change_type'].lower():
            report_lines.append(f"[{verdict['risk_level']}] [{verdict['change_type']}] [{verdict['explaination']}]"
            )
    return "\n".join(report_lines) if report_lines else "No Material difference found!."
agent=create_agent(model=llm,tools=[compare_documents],
                   system_prompt="You are a contract comparison agent. Use the compare_documents tool "
              "whenever the user provides two document paths, then return the FULL "
              "report EXACTLY as given by the tool, verbatim, with no rewording, "
              "paraphrasing, or summarizing of any field.",
)























