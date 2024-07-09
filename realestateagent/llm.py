import logging
from sentence_transformers import SentenceTransformer
from mistralai.client import MistralClient
from qdrant_client import QdrantClient
from llama_index.llms.mistralai import MistralAI

# Initialize MistralAI and QdrantClient instances
api_key = "XWVBd37gZ0dE1KdYgqTLaWIvJ2kvMHSy"
llm = MistralAI(api_key=api_key)
qdrant_client = QdrantClient(
    url="https://28c99715-afa1-4ccd-96d1-eb88c57c2523.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="X94ZyHIl-vwHdDwBvrnzSTiApcJGRyiG0sCuO6JHbO8jSLYV3EfW3A",
)

# Initialize SentenceTransformer encoder
# encoder = SentenceTransformer("realestateagent/model")


# Initialize MistralClient for chat functionality
def run_mistral(
    user_message, model="mistral-large-latest", system=None, output_format=None
):
    # logging.info("Calling run_mistral function")
    client = MistralClient(api_key=api_key)
    if system is None:
        messages = [{"role": "user", "content": user_message}]
        chat_response = client.chat(model=model, messages=messages)

    elif output_format is not None:
        chat_response = client.chat(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )
    else:
        chat_response = client.chat(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
    return chat_response.choices[0].message.content


def run_codestral(user_message):
    client = MistralClient(api_key=api_key)
    messages = [
        {"role": "user", "content": user_message},
    ]

    chat_response = client.chat(model="codestral-latest", messages=messages)
    return chat_response.choices[0].message.content


def execute_pandas_query(query, data):
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
            this is the description of the important columns:
            * brokered by (categorically encoded agency/broker)
            * status (Housing status - a. ready for sale or b. ready to build)
            * price (Housing price, it is either the current listing price or recently sold price if the house is sold recently)
            * bed (# of beds)
            * bath (# of bathrooms)
            * acre_lot (Property / Land size in acres)
            * street (categorically encoded street address)
            * city (city name)
            * state (state name)
            * zip_code (postal code of the area)
            * house_size (house area/size/living space in square feet)
            * prev_sold_date (Previously sold date)
            
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

    try:
        # logging.info(f"query: {query}")
        return f"query: {query} \nExpression: {expression}\nOutput: {eval(expression)}"
    except Exception as e:
        logging.info(f">>>>>> cought exception : {e}")
        new_prompt = """
        the execution of this expression ({expression}) failed with the following error:
        ({e})
        Try rewriting the expression to avoid errors. while staying within the constraints of the query ({query}).
        """.format(
            expression=expression, e=e, query=query
        )
        expression = run_codestral(new_prompt)
        try:
            return (
                f"query: {query} \nExpression: {expression}\nOutput: {eval(expression)}"
            )
        except Exception as e:
            logging.info(f">>>>>> cought exception : {e}")
            return f"query: {query} \nExpression: {expression}\nOutput: {e}"
