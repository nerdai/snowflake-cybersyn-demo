from typing import Any, Generator

import streamlit as st
from llama_index.core.llms import ChatMessage, ChatResponseGen
from llama_index.llms.openai import OpenAI


def _llama_index_stream_wrapper(
    llama_index_stream: ChatResponseGen,
) -> Generator[str, Any, Any]:
    for chunk in llama_index_stream:
        yield chunk.delta


st.title("ChatGPT-like clone")

llm = OpenAI(model="gpt-4o-mini")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = llm.stream_chat(
            messages=[
                ChatMessage(role=m["role"], content=m["content"])
                for m in st.session_state.messages
            ]
        )
        response = st.write_stream(_llama_index_stream_wrapper(stream))
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
