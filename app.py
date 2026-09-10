import streamlit as st
import tempfile
from agent import agent

st.title("Contract / Policy Comparator Agent")
st.write("Upload two version of policy or contract to see what changed and what is risky")

file_a=st.file_uploader('Document A (older version)',type='pdf')
file_b=st.file_uploader('Document B (newer version)',type='pdf')

if file_a and file_b and st.button("Compare"):
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_a:
        tmp_a.write(file_a.read())
        path_a=tmp_a.name
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_b:
        tmp_b.write(file_b.read())
        path_b=tmp_b.name

    with st.spinner("Comparing.."):
        result=agent.invoke({
            'messages':[{'role':'user','content':f" Compare {path_a} and {path_b}"}]
        })
        final_message=result['messages'][-1].content

    st.subheader("Comparison Report")
    st.write(final_message)









