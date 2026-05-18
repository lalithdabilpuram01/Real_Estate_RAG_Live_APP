import os
import time

import streamlit as st
from rag import process_urls, generate_answer




# Set up the main title of the Streamlit web application
st.title("Real Estate Research Tool")

# Initialize the session state to securely store the API key if it's not already present
if 'api_key_input' not in st.session_state:
    st.session_state['api_key_input'] = ''

# Create a password-type text input field in the sidebar for the user to enter their Groq API key
api_key = st.sidebar.text_input("Enter your groq api key here", type= "password",value=st.session_state.get('api_key_input',''))



# If the user has provided an API key, save it and proceed to load the rest of the UI
if api_key:
    # Remove any extra whitespace or quotes from the input
    cleaned_key = api_key.strip().strip("'\"")
    # Save the cleaned key to the session state and set it as an environment variable
    st.session_state['api_key_input'] = cleaned_key
    os.environ["GROQ_API_KEY"] = api_key
    st.sidebar.success("API Key Active")

    # Input fields for up to three target URLs to scrape real estate data from
    url1 = st.sidebar.text_input("URL 1")
    url2 = st.sidebar.text_input("URL 2")
    url3 = st.sidebar.text_input("URL 3")

    # An empty placeholder to display dynamic status updates to the user
    placehoder = st.empty()

    # Button to trigger the data extraction and vectorization process
    process_url_button = st.sidebar.button("Process URLs")

# When the user clicks the process button
    if process_url_button:
        # Collect only the URLs that the user has actually filled out
        urls = [url for url in (url1,url2,url3) if url != '' ]

        # Validate that at least one URL was provided
        if len(urls)==0:
            placehoder.text("You must provide at least one valid url")

        else:
            # Process the URLs and continuously update the placeholder text with the current status
            for status in process_urls(urls):
                placehoder.text(status)

    # Input field for the user to ask questions about the processed real estate data
    query = placehoder.text_input("Question")

# If the user submits a question
    if query :
        try :
            # Generate the answer based on the query and retrieve the sources used
            answer, sources = generate_answer(query)
            
            # Display the generated answer
            st.header("Answer:")
            st.write(answer)

            # If sources were found, display them as well
            if sources:
                st.subheader("Sources:")
                # Split the sources string by newline to print them out cleanly
                for source in sources.split('\n'):
                    st.write(source)

        except RuntimeError as e:
            # Catch errors typically caused by querying before URLs have been processed
            placehoder.text("You must process urls first")

# Provide an option to clear the active API key and reset the application state
    clear_api_key_button = st.sidebar.button("Clear API Key")
    if clear_api_key_button :
        # Remove the API key from environment variables
        del os.environ["GROQ_API_KEY"]
        del cleaned_key , api_key

        # Clear Streamlit's session state and all cached data to fully reset the app
        st.session_state.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Notify the user that the key has been cleared
        success_msg = st.success("cleared api key")

        # Briefly wait before reloading the application page
        if success_msg :
            time.sleep(5)
            success_msg.empty()
            st.rerun()
