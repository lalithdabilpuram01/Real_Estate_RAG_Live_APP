import os
import time

import streamlit as st
from rag import process_urls, generate_answer




st.title("Real Estate Research Tool")

if 'api_key_input' not in st.session_state:
    st.session_state['api_key_input'] = ''

api_key = st.sidebar.text_input("Enter your groq api key here", type= "password",value=st.session_state.get('api_key_input',''))



if api_key:
    cleaned_key = api_key.strip().strip("'\"")
    st.session_state['api_key_input'] = cleaned_key
    os.environ["GROQ_API_KEY"] = api_key
    st.sidebar.success("API Key Active")




    url1 = st.sidebar.text_input("URL 1")
    url2 = st.sidebar.text_input("URL 2")
    url3 = st.sidebar.text_input("URL 3")

    placehoder = st.empty()

    process_url_button = st.sidebar.button("Process URLs")

    if process_url_button:
        urls = [url for url in (url1,url2,url3) if url != '' ]

        if len(urls)==0:
            placehoder.text("You must provide at least one valid url")

        else:
            for status in process_urls(urls):
                placehoder.text(status)

    query = placehoder.text_input("Question")

    if query :

        try :
            answer, sources = generate_answer(query)
            st.header("Answer:")
            st.write(answer)

            if sources:
                st.subheader("Sources:")
                for source in sources.split('\n'):
                    st.write(source)

        except RuntimeError as e:
            placehoder.text("You must process urls first")

    clear_api_key_button = st.sidebar.button("Clear API Key")
    if clear_api_key_button :
        del os.environ["GROQ_API_KEY"]
        del cleaned_key , api_key

        st.session_state.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        success_msg = st.success("cleared api key")

        if success_msg :
            time.sleep(5)
            success_msg.empty()
            st.rerun()
