import logging
import sys
import json
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from mistralai.client import MistralClient
from qdrant_client import QdrantClient
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.mistralai import MistralAI
from llama_index.core import PromptTemplate
from streamlit_chat import message
import ast

# Initialize logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
data = pd.read_csv("realestateagent/data_prices_cleaned.csv")
# Initialize Streamlit page configuration
st.set_page_config(page_title="Neuralytics - Real Estate", page_icon=":robot:")
st.header("Neuralytics - Real Estate")
st.markdown(":red[Neuron-Q]")

# Initialize session state variables
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# Initialize MistralAI and QdrantClient instances
api_key = "XWVBd37gZ0dE1KdYgqTLaWIvJ2kvMHSy"
llm = MistralAI(api_key=api_key)
qdrant_client = QdrantClient(
    url="https://28c99715-afa1-4ccd-96d1-eb88c57c2523.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="X94ZyHIl-vwHdDwBvrnzSTiApcJGRyiG0sCuO6JHbO8jSLYV3EfW3A",
)

# Initialize SentenceTransformer encoder
encoder = SentenceTransformer("realestateagent/model")

# Set MistralAI instance in settings
from llama_index.core import Settings

Settings.llm = llm


# Initialize MistralClient for chat functionality
def run_mistral(user_message, model="mistral-medium", system=None):
    # logging.info("Calling run_mistral function")
    client = MistralClient(api_key=api_key)
    if system is None:
        messages = [{"role": "user", "content": user_message}]
        chat_response = client.chat(model=model, messages=messages)

    else:
        chat_response = client.chat(
            model="open-mixtral-8x7b",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )

    return chat_response.choices[0].message.content


def run_codestral(user_message):
    client = MistralClient(api_key=api_key)
    messages = [{"role": "user", "content": user_message}]
    chat_response = client.chat(model="codestral-latest", messages=messages)
    return chat_response.choices[0].message.content


# Function to classify user intent based on message content
def classify_user_intent(message):
    # logging.info("Calling classify_user_intent function")
    prompt = """
        You are a real estate data assistant. Your task is to determine if the user's message indicates an intention to retrieve data or analyze data related to real estate.
        Categorize the message into one of the following predefined categories:
        
        retrieve data
        analyze data
        
        
        
        You will only respond with the predefined category. Do not include the word "Category". Do not provide explanations or notes. 
        
        ####
        Here are some examples:
        
        
        Message: What's the current rental rate for a 2-bedroom apartment in downtown Chicago?
        Category: analyse data
        Message: Can you analyze the factors affecting property values in suburban areas?
        Category: analyze data
        Message: how many appartements for sale are there in ariana ?
        Category: analyze data
        Message: I want to buy a house in Ariana.
        Category: retrieve data
        Message: I have a budget under 200000 dt and I want to rent an appartment . give me 5 Options
        Category: retrieve data
        Message I'm looking for a 3 bedroom apartment.
        Category: retrieve data
        ###
    
        <<<
        Message: {}
        >>>
        """.format(
        message
    )
    return run_mistral(prompt)


def recommand_chart(query, data):
    prompt = """
            You are a data analyst. Your task is to provide the best chart to visualize the data based on the user's query.
            You have access to the following charts : 
            - bar chart
            - line chart
            - scatter chart
            return a dictionnary with this format :
            {
                'chart_type': 'bar',
                'x':Column name or key associated to the x-axis data,
                'y':Column name(s) or key(s) associated to the y-axis data,
                'x_label':The label for the x-axis,
                'y_label':The label for the y-axis,
            }
            >>>>
            query :{}
            data:{}
    """.format(
        query, data
    )
    return run_mistral(prompt)


def execute_pandas(query, data=data):
    try:
        new_prompt = PromptTemplate(
            """\
            You are working with a pandas dataframe in Python.
            Important : In Pandas, the correct way to combine multiple conditions is to use parentheses around each condition and the & operator between them.
            Example : df.loc[(df['location'].str.contains('Ariana')) & (df['transaction'] == 'location')].price.mean()
            The name of the dataframe is `df`.
            always return the full row when the result is to select data.
            This is the result of `print(df.head())`:
            {df_str}
            this is the description of the important columns:
            * `location`: the location of the property (State,city)
            * `transaction`: the type of transaction ['sale' OR 'rent']
            * `price`: the price of the property the currency is DT ,type(price) is float
            * `contact`: the contact of the seller
            * `category`: the category of the property ['Appartements', 'Maisons et Villas', 'Terrains et Fermes',
                'Magasins, Commerces et Locaux industriels', 'Autres Immobiliers',
                'Bureaux et Plateaux', 'Colocations', 'Locations de vacances']
            * `chambres`: the number of rooms
            * `salle_de_bain`: the number of bathrooms
            * `descriptions`: the description of the property which can be used to filter the number of rooms (exp : S+1 is a one bedroom apartment)
            * `profiles`: the link to the profile of the agency on tayara.tn

            Follow these instructions:
            {instruction_str}
            Query: {query_str}

            Expression: 
            """
        )

        query_engine = PandasQueryEngine(df=data, verbose=False)

        # print(prompts["response_synthesis_prompt"])
        query_engine.update_prompts({"pandas_prompt": new_prompt})

        return (
            query_engine.query(query)
            if type(query) == str
            else query_engine.query(query["inputs"]["text"])
        )
    except:
        logging.info(">>>>>> cought exception")


def execute_pandas_query(query, data=data):
    try:
        df = data.copy()
        prompt = f"""
            You are working with a pandas dataframe in Python.
            The name of the dataframe is `df`.
            This is the result of `print(df.head())`:
            {df.head()}
            This is the result of `print(df.info())`:
            {df.info()}
            This is the result of `print(df.describe())`:
            {df.describe()}
            This is the result of `print(df.columns)`:
            {df.columns}
            this is the description of the important columns:
            * `location`: the location of the property (State,city)
            * `transaction`: the type of transaction ['sale' OR 'rent']
            * `price`: the price of the property the currency is DT ,type(price) is float
            * `contact`: the contact of the seller
            * `category`: the category of the property ['Appartements', 'Maisons et Villas', 'Terrains et Fermes',
                'Magasins, Commerces et Locaux industriels', 'Autres Immobiliers',
                'Bureaux et Plateaux', 'Colocations', 'Locations de vacances']
            * `chambres`: the number of rooms
            * `salle_de_bain`: the number of bathrooms
            * `descriptions`: the description of the property which can be used to filter the number of rooms (exp : S+1 is a one bedroom apartment)
            * `profiles`: the link to the profile of the agency on tayara.tn
            
            Query: {query}

            Expression: 
        """
        system = """
            Follow these instructions:
            1. Convert the query to executable Python code using Pandas.\n
            2. The final line of code should be a Python expression that can be called with the `eval()` function.\n
            3. The code should represent a solution to the query.\n
            4. PRINT ONLY THE EXPRESSION.\n
            5. Do not quote the expression.\n
            6. Do not provide any notes or explenations.\n
        """
        expression = run_mistral(prompt, system=system)
        # logging.info(f"query: {query}")
        return f"query: {query} \nExpression: {expression}\nOutput: {eval(expression)}"
    except:

        logging.info(">>>>>> cought exception")


# Function to generate questions based on the query for data analysis
def analyse(query, data=data):
    # logging.info("Calling analyse function")
    prompt = """
        You are a senior data analyst. Your task is to generate questions to ask about the data in order to perfectly answer the query.
        You will generate a maximum of 10 questions depending on the complexity of the query.
        The data you will use is the real estate prices in Tunisia.
        This is the result of `print(df.head())`:
            {}
            This is the result of `print(df.info())`:
            {}
            This is the result of `print(df.describe())`:
            {}
            This is the result of `print(df.columns)`:
            {}
            this is the description of the important columns:
            * `location`: the location of the property (State,city)
            * `transaction`: the type of transaction ['sale' OR 'rent']
            * `price`: the price of the property the currency is DT ,type(price) is float
            * `category`: the category of the property ['Appartements', 'Maisons et Villas', 'Terrains et Fermes',
                'Magasins, Commer
        Stricktly return a List of questions. Do not include the word "question". Do not provide explanations or notes.
        <<<>>>
        Here are some examples:
        - Query: Can you provide the market trend in Ariana?
          Questions: ['What are the average property prices in Ariana for sale and rent?', 'What is the trend in the number of property listings in Ariana over time?', 'What are the most common property categories in Ariana?', 'How do the average prices vary by the number of rooms in Ariana?', 'Are there any notable differences in property prices based on the city areas within Ariana?']
        
        - Query: What are the market trends?
          Questions: ['What are the overall market trends, considering both sale and rent transactions?', 'How have property prices trended over time?', 'What factors are influencing the current market conditions?', 'Can you provide insights into the demand-supply dynamics of properties?']
        <<<>>>
        Query: {}
        """.format(
        data.head(), data.info(), data.describe(), data.columns, query
    )
    system = """
            Stricktly return a List of questions. Do not include the word "question". Do not provide explanations or notes.
        """
    return run_mistral(prompt, system=system)


def deep_analyse(questions):
    results = []
    for question in questions:
        # output = execute_pandas(question)
        output = execute_pandas_query(question)
        results.append({"Question": question, "pandas_output": output})
        # logging.info(recommand_chart(question, output))
    return results


# Function to synthesize a response based on retrieved and generated data
def synthesize_retrival_response(data, query, intent, retrieved="None"):
    # logging.info("Calling synthesize_response function")
    prompt = """
        Your task is to generate a response to the user's query.
        You are given a pandas output, semantically retrieved data, and the user query.
        Synthesize a response to the query based on the cross combination of the two sources.
        Ensure clarity, completeness, and relevance to the user Query.
        Use markdowns to format your response.
        Don't mention that you are using pandas in your response.
        Pay attention to the user intent: {}.
        
        # Query:
        {}
        
        # pandas output:
        {}
        
        # retrieved data:
        {}
        """.format(
        intent, query, data, retrieved
    )

    return run_mistral(prompt)


def synthesize_ranalyse_response(query, questions):
    # logging.info("Calling synthesize_response function")
    prompt = """
        
        

        ## Task Details:
        You will be provided with:
        - A specific query (`Query:`) that outlines the problem to be solved or the question to be answered.
        - Intermediary questions (`Questions and pandas outputs:`) along with their corresponding outputs generated using pandas.
        - You can use your knowledge outside of the provided data to further refine your analysis and provide more comprehensive insights.
        ## Requirements:
        - Don't mention that you are using pandas in your response.
        - Utilize the pandas outputs provided to generate a final analysis that addresses the query effectively.
        - Ensure your response is well-structured and formatted using Markdown.
        - Focus on clarity and completeness in your analysis, providing meaningful interpretations and insights based on the data.
        - Don't repeat the pandas outputs in your response, use table markdowns to provide a clear and concise summary of the data.
        - the currency is Tunisian Dinar.
        - always use metric system
        - don't include questions that does not have a correct pandas output.
        # Query:
        {}
        
        # Questions and pandas outputs:
        {}
        
        
        """.format(
        query, questions
    )
    system = """
        As an experienced data analyst, your task is to analyze provided data to derive insights and answer a specific query. 
        Your analysis should be clear, comprehensive, and directly relevant to the given problem statement.
    """
    return run_mistral(prompt, system=system)


# Function to retrieve data from Qdrant based on user input
def generate_response_retrieve_data(input_str: str) -> str:
    # logging.info("Calling generate_response_retrieve_data function")
    hits = qdrant_client.search(
        collection_name="RealEstateSementic",
        query_vector=encoder.encode(input_str).tolist(),
        limit=3,
    )
    return "\n".join([str(hit.payload) for hit in hits])


# Main function to generate response based on user input
def generate_response(query: json):
    # logging.info("Calling generate_response function")
    retrieved = "None"
    query_class = classify_user_intent(query["inputs"]["text"])
    logging.info(f">>>>> query class: {query_class}")

    if query_class == "retrieve data":
        retrieved = generate_response_retrieve_data(query["inputs"]["text"])
        logging.info(f">>>>> retrieved data: {retrieved}")
        generated_text = execute_pandas(query, data)

        response = synthesize_retrival_response(
            generated_text, query["inputs"]["text"], query_class, retrieved
        )
    else:
        analyse_questions = analyse(query["inputs"]["text"])
        logging.info(f">>>>> analyse_questions: {analyse_questions}")
        Questions = ast.literal_eval(analyse_questions)
        res = deep_analyse(Questions)
        Questions_str = "\n".join(str(item) for item in res)
        response = synthesize_ranalyse_response(query["inputs"]["text"], Questions_str)
        logging.info(f">>> Questions: {type(Questions)}")

    # Assuming the Pandas dataframe `data` is global and accessible

    return {"generated_text": str(response)}


# Function to handle user input and display responses
def handle_user_input():
    # logging.info("Handling user input")
    user_input = st.text_input("You: ")

    if user_input:
        output = generate_response(
            {
                "inputs": {
                    "past_user_inputs": st.session_state.past,
                    "generated_responses": st.session_state.generated,
                    "text": user_input,
                },
                "parameters": {"repetition_penalty": 1.33},
            }
        )

        st.session_state.generated.append(output["generated_text"])
        st.session_state.past.append(user_input)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


# Run the application
handle_user_input()
