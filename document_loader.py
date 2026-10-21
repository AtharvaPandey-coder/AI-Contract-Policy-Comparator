from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_chunk(pdfpath:str)->str:

    """
    Loads a pdf and split it into claused sized chunks.
    chunk_size=800 roughly matches one contract clause/paragraph.
    chunk_overlap=100 keeps content from bleeding across a clause boundary.
    """
    loader=PyPDFLoader(pdfpath)
    docs=loader.load()

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=0,
        separators=["\n\n","\n"," "]#tries paragraph breaks first and then sentences.
    )
    chunks=splitter.split_documents(docs)

    for i,chunk in enumerate(chunks):
        chunk.metadata['chunk_id']=i
    return chunks        







