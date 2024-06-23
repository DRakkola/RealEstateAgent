import json
import logging
import sys

from llama_index.core import Settings
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.mistralai import MistralAI
from llama_index.core import PromptTemplate
import os

llm = MistralAI(api_key="XWVBd37gZ0dE1KdYgqTLaWIvJ2kvMHSy")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logging.info(os.chdir("."))
data = pd.read_csv("realestateagent/data_prices_cleaned.csv")
data.drop(columns=["Unnamed: 0", "id"], axis=1, inplace=True)
Settings.llm = llm


def generate_response(query: json):
    new_prompt = PromptTemplate(
        """\
You are working with a pandas dataframe in Python.
Important : In Pandas, the correct way to combine multiple conditions is to use parentheses around each condition and the & operator between them.
Example : df.loc[(df['location'].str.contains('Ariana')) & (df['transaction'] == 'location')].price.mean()
The name of the dataframe is `df`.
This is the result of `print(df.head())`:
{df_str}
this is the description of the important columns:
* `location`: the location of the property (State,city)
* `transaction`: the type of transaction ['sale' OR 'rent']
* `price`: the price of the property the currency is DT ,type(price) is float
* `contact`: the contact of the seller
* `chambres`: the number of rooms
* `descriptions`: the description of the property which can be used to filter the number of rooms (exp : S+1 is a one bedroom apartment)

in each call, make sure to return the following columns `location`, `transaction`, `price`, `contact`,`descriptions` 
Follow these instructions:
{instruction_str}
Query: {query_str}

Expression: """
    )
    new_synthesis_prompt = PromptTemplate(
        """\
        You are a Real Estate Agent, and you are helping a customer to find the best offers.
        Important : Answer in the same language as the customer and priotize the french language.You only answer based on Pandas Output atherwise say that you don't know.
        Given an input question, synthesize a response based on the query results.
        The response must be easy to understand .
        when responding huminize (price,descriptions and contact) each row into one bullet point.
        Don't repeat the pandas output, humanize the response.
        when talking about price use the currency DT.
        if the pandas output does not match the query, focus on answering the query.
        Query: {query_str}\n\n
        Pandas Instructions (optional):\n{pandas_instructions}\n\n
        Pandas Output: {pandas_output}\n\n
        Response:
        """
    )
    query_engine = PandasQueryEngine(df=data, synthesize_response=True, verbose=True)

    # print(prompts["response_synthesis_prompt"])
    query_engine.update_prompts(
        {"pandas_prompt": new_prompt, "response_synthesis_prompt": new_synthesis_prompt}
    )
    recent_user_inputs = query["inputs"]["past_user_inputs"][-4:]
    recent_responses = query["inputs"]["generated_responses"][-4:]

    query_text = (
        f"{query['inputs']['text']} "
        f"previous queries: {' '.join(recent_user_inputs)} "
        f"previous responses: {' '.join(recent_responses)}"
    )
    generated_text = query_engine.query(query["inputs"]["text"])
    return {"generated_text": str(generated_text)}


import streamlit as st
from streamlit_chat import message


st.set_page_config(page_title="RealEstate Chat - Demo", page_icon=":robot:")


st.header("RealEstate Chat - Demo")
st.markdown("[Github](https://github.com/ai-yash/st-chat)")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def query(payload):
    response = generate_response(payload)
    return response


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = query(
        {
            "inputs": {
                "past_user_inputs": st.session_state.past,
                "generated_responses": st.session_state.generated,
                "text": user_input,
            },
            "parameters": {"repetition_penalty": 1.33},
        }
    )

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output["generated_text"])

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
