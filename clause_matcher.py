from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def build_index(chunks):
    """Embed all chunks of one document and stores them into a FAISS index."""
    return FAISS.from_documents(chunks,embedding)

def align_clause(chunks_a,index_b,top_k=1):

    """
    For every clause in Docuemnt A ,finds its closest semantic match in Docuemnt B.
    This is what lets the agent ,compare a reworded clause,not just identical text.
    Returns a list of (chunks_a,best_match_in_b,similarity_search). 
    """
    pairs=[]
    for chunk in chunks_a:
        results=index_b.similarity_search_with_score(chunk.page_content,k=top_k)
        best_match,score=results[0]
        pairs.append({
            'clause_a':chunk.page_content,
            'clause_b':best_match.page_content,
            'score':score # lower=more similar (FAISS returns L2 distance)
        })
    return pairs
