import random
import time
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
import openai


API_INFOS = [
    {
        "endpoints": "https://conversationhubeastus.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://conversationhubeastus2.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://conversationhubsouthcentralus.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://conversationhubwestus.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://conversationhubwestus3.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://readineastus.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    },
    {
        "endpoints": "https://readineastus2.openai.azure.com/",
        "speed": 150,
        "model": "gpt-4o"
    }
]



class Openai():
    def __init__(
            self,
            apis,
            identity_id="18731b9d-0488-49be-bb05-6ccb08f78cf3",

    ):
        self.identity_id = identity_id
        flag = True
        while flag:
            try:
                self.token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(managed_identity_client_id=self.identity_id),
                    "https://cognitiveservices.azure.com/.default"
                )
                flag = False
                break
            except:
                continue
        # print(f"apis:{apis}", flush=True)
        self.clients_weight = [apis[i]['speed'] for i in range(len(apis))]
        weight_sum = sum(self.clients_weight)
        for i in range(len(self.clients_weight)):
            self.clients_weight[i] /= weight_sum


        selected_api = random.choices(apis, weights=self.clients_weight, k=1)[0]

        self.client = AzureOpenAI(
            azure_endpoint=selected_api['endpoints'],
            azure_ad_token_provider=self.token_provider,
            api_version="2024-04-01-preview",
            max_retries=0,
        )
        self.model = selected_api['model']

    def get_image_response(self, content, image1, image2, system="You are a helpful assistant.", max_tokens=2048, client_index = None):

        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')   
                 
        base64_image1 = encode_image(image1)    
        base64_image2 = encode_image(image2)        
        messages = [
            {"role": "system", "content": f"{system}"},
            {"role": "user", "content": [
                {"type": "text", "text": f"{content}\n"},
                {
                    "image":base64_image1
                },
                {
                    "image":base64_image2
                }
            ]
             },
        ]


        client = self.client
        model = self.model

        max_retry = 5
        cur_retry = 0
        while cur_retry <= max_retry:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                )

                # client.chat.completions.with_raw_response
                results = completion.choices[0].message.content
                return results
            except openai.RateLimitError as e:
                time.sleep(1)
            except Exception as e:
                print(e)
                cur_retry += 1
        return ""
    
    def get_response(self, messages, temperature=0, max_tokens=2048, top_p=0.95):


        client = self.client
        model = self.model

        max_retry = 5
        cur_retry = 0
        while cur_retry <= max_retry:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0,
                    top_p=top_p,
                    stop=None
                )

                # client.chat.completions.with_raw_response
                results = completion.choices[0].message.content
                return results
            except openai.RateLimitError as e:
                time.sleep(1)
            except Exception as e:
                print(e)
                cur_retry += 1
        return ""

    def get_response_raw(self, messages, temperature=0, max_tokens=2048, top_p=0.95):


        client = self.client
        model = self.model

        max_retry = 5
        cur_retry = 0
        while cur_retry <= max_retry:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0,
                    top_p=top_p,
                    stop=None
                )

                # client.chat.completions.with_raw_response
                # results = completion.choices[0].message.content
                results = completion
                return results
            except openai.RateLimitError as e:
                time.sleep(1)
            except Exception as e:
                print(e)
                cur_retry += 1
        return ""
       
    def call(self, content, client_index = None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": f"{content}\n"},
            ]
             },
        ]

        client = self.client
        model = self.model

        max_retry = 5
        cur_retry = 0
        while cur_retry <= max_retry:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=64,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                )

                # client.chat.completions.with_raw_response
                results = completion.choices[0].message.content
                return results
            except openai.RateLimitError as e:
                time.sleep(1)
            except Exception as e:
                print(e)
                cur_retry += 1
        return ""

if __name__ == '__main__':
    oai_clients = Openai(
        apis=API_INFOS
    )
    # res = oai_clients.call("你是gpt几？")
    content = "你是gpt几？"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": f"{content}\n"},
        ]
            },
    ]    
    # res = oai_clients.get_response(messages)
    res = oai_clients.get_response_raw(messages)
    print(res)
