from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

import requests
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import logging
import httpx
from httpcore._exceptions import ReadTimeout
from requests.exceptions import ConnectionError
# from openai import OpenAI
# import timeout

class RateLimitReached(Exception):
    pass

class OfficialError(Exception):
    pass

class RecoverableError(Exception):
    pass

class KeysBusyError(Exception):
    pass

class ChatGPTWrapper:
    def __init__(self, api_key='', proxies=None) -> None:
        # Set the request parameters
        self.url = 'https://api.openai.com/v1/chat/completions'
        # Set the header
        self.header = {
            "Content-Type": "application/json",
            "Authorization": 'Bearer {}'.format(api_key)
        }
        self.proxies = proxies

    @retry(wait=wait_random_exponential(min=1, max=60), retry=retry_if_exception_type((RateLimitReached, RecoverableError, OfficialError, ConnectionError)))
    def call(self, messages, **kwargs):
        query = {
            "model": "gpt-3.5-turbo",
            "messages": messages
        }
        query.update(kwargs)

        # Make the request
        if self.proxies:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query), proxies=self.proxies)
        else:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query))
        response = response.json()
        if 'error' in response and 'Rate limit reached' in response['error']['message']:
            raise RateLimitReached()
        elif 'choices' in response:
            return response
        else:
            if 'error' in response:
                print(response['error']['message'])
                if response['error']['message'] == 'The server had an error while processing your request. Sorry about that!':
                    raise RecoverableError(response['error']['message'])
                else:
                    raise OfficialError(response['error']['message'])
            else:
                raise Exception('Unknown error occured. Json: {}'.format(response))
            
class AIONEWrapper:
    def __init__(self, api_key='', base_url = "https://api.platform.a15t.com/v1") -> None:
        
        # Set the base URL and the API key for authorization
        self.url = f"{base_url}/chat/completions"
        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    # def call(self, messages, tools, tool_choice, **kwargs):
    def call(self, messages,  **kwargs):
        # Prepare the query for the request
        query = {
            # "model": "openai/gpt-4-0613",  # Set the model name (adjust if necessary)
            "model": "openai/gpt-4o-mini-2024-07-18",
            "messages": messages,
        }
        query.update(kwargs)  # Add any additional parameters

        try:
            with httpx.Client(http2=True, timeout=20) as client:
                response = client.post(self.url, headers=self.header, json=query)
                response.raise_for_status()  # Raise an error for bad responses
                response_data = response.json()
                return response_data

        except ReadTimeout:
            print("Request timed out. Try again later.")
            return "None"
        # except httpx.HTTPStatusError as e:
        #     print(f"HTTP error occurred: {e}")
        #     return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "None"
            
class ArceeWrapper:
    def __init__(self, api_key='', base_url="https://models.arcee.ai/v1") -> None:
        # Set the base URL and the API key for authorization
        self.url = f"{base_url}/chat/completions"
        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    # def call(self, messages, tools, tool_choice, **kwargs):
    def call(self, messages,  **kwargs):
        # Prepare the query for the request
        query = {
            "model": "caller",  # Set the model name (adjust if necessary)
            "messages": messages,
            # "tools": tools,
            # "tool_choice": tool_choice
        }
        query.update(kwargs)  # Add any additional parameters

        try:
            with httpx.Client(http2=True, timeout=10) as client:
                response = client.post(self.url, headers=self.header, json=query)
                response.raise_for_status()  # Raise an error for bad responses
                response_data = response.json()
                return response_data

        except ReadTimeout:
            print("Request timed out. Try again later.")
            return " "
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return ""
            
class GPT4Wrapper(ChatGPTWrapper):
    @retry(wait=wait_random_exponential(min=1, max=60), retry=retry_if_exception_type((RateLimitReached, RecoverableError, OfficialError, ConnectionError)))
    def call(self, messages, **kwargs):
        query = {
            "model": "gpt-4-0314",
            "messages": messages
        }
        query.update(kwargs)

        # Make the request
        if self.proxies:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query), proxies=self.proxies)
        else:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query))        
        response = response.json()
        if 'error' in response and 'Rate limit reached' in response['error']['message']:
            raise RateLimitReached()
        elif 'choices' in response:
            return response
        else:
            if 'error' in response:
                print(response['error']['message'])
                if response['error']['message'] == 'The server had an error while processing your request. Sorry about that!':
                    raise RecoverableError(response['error']['message'])
                else:
                    raise OfficialError(response['error']['message'])
            else:
                raise Exception('Unknown error occured. Json: {}'.format(response))
                
class DavinciWrapper:
    def __init__(self, api_key='') -> None:
        # Set the request parameters
        self.url = 'https://api.openai.com/v1/completions'
        # Set the header
        self.header = {
            "Content-Type": "application/json",
            "Authorization": 'Bearer {}'.format(api_key)
        }

    @retry(wait=wait_random_exponential(min=1, max=60), retry=retry_if_exception_type((RateLimitReached, RecoverableError, OfficialError, ConnectionError)))
    def call(self, messages, **kwargs):
        # messages to prompt
        prompt = ''
        for message in messages:
            prompt += message['role'] + ': ' + message['content'] + '\n'

        query = {
            "model": "davinci",
            "prompt": prompt
        }
        query.update(kwargs)

        # Make the request
        response = requests.post(self.url, headers=self.header, data=json.dumps(query))
        response = response.json()
        if 'error' in response and 'Rate limit reached' in response['error']['message']:
            raise RateLimitReached()
        elif 'choices' in response:
            return response
        else:
            if 'error' in response:
                print(response['error']['message'])
                if response['error']['message'] == 'The server had an error while processing your request. Sorry about that!':
                    raise RecoverableError(response['error']['message'])
                else:
                    raise OfficialError(response['error']['message'])
            else:
                raise Exception('Unknown error occured. Json: {}'.format(response))

class GPT4Wrapper(ChatGPTWrapper):
    @retry(wait=wait_random_exponential(min=1, max=60), retry=retry_if_exception_type((RateLimitReached, RecoverableError, OfficialError, ConnectionError)))
    def call(self, messages, **kwargs):
        query = {
            "model": "gpt-4-0314",
            "messages": messages
        }
        query.update(kwargs)

        # Make the request
        if self.proxies:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query), proxies=self.proxies)
        else:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query))        
        response = response.json()
        if 'error' in response and 'Rate limit reached' in response['error']['message']:
            raise RateLimitReached()
        elif 'choices' in response:
            return response
        else:
            if 'error' in response:
                print(response['error']['message'])
                if response['error']['message'] == 'The server had an error while processing your request. Sorry about that!':
                    raise RecoverableError(response['error']['message'])
                else:
                    raise OfficialError(response['error']['message'])
            else:
                raise Exception('Unknown error occured. Json: {}'.format(response))

