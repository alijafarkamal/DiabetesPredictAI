import re
import subprocess
import sys
from .base import LLMTool
class CodeEngine(LLMTool):
    name: str = "Code Generation and Execution Tool"
    description: str = "A coding tool that can take a prompt and generate executable Python code. It parses and executes the code. Returns the code and the error if the code execution fails."
    arg: str = "A single string parameter describing the coding task."
    def parse_and_exec_code(self, response: str):
        result = re.search(r'```python\s*([\s\S]*?)\s*```', response)
        if not result:
            return "No Python code block found", "Failed to extract code"
        code_string = result.group(1)
        if "pip install" in code_string.split("\n")[0]:
            print("Requires PIP package installations")
            packages = code_string.split("\n")[0].split("pip install")[-1].strip()
            if "," in packages:
                packages = packages.split(",")
            elif " " in packages:
                packages = packages.split(" ")
            else:
                packages = [packages]
            print(f"Installing packages: {packages}")
            for package in packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("Executing main code...")
        try:
            exec(code_string)
        except Exception as e:
            print(f"Error executing generated code: {e}")
            return code_string, e
        return code_string, None
    #def generate_code(self, prompt):
    #    response = self.client.chat.completions.create(
    #        model="gpt-4o", # DEFAULT TO GPT-4o , BUT MAKE IT VARIABLE W/ OPEN ROUTER MODELS 
    #        messages=[
    #            {"role": "system", "content": "You are a Python code generator. Respond only with executable Python code, no explanations or comments except for required pip installations at the top. Return the code within ```python and ``` strings. The first line should be commented out pip install statement"},
    #            {"role": "user", "content": f"Generate Python code to {prompt}. If you need to use any external libraries, include a comment at the top of the code listing the required pip installations."}
    #        ],
    #        max_tokens=4000, temperature=0.7)
    #    response = response.choices[0].message.content
    #    code, error = self.parse_and_exec_code(response)
    #    return code, error     
    def generate_code(self, prompt):
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        model_name = os.environ.get("MODEL_NAME", "gpt-4o")  # Default to gpt-4o if MODEL_NAME is not set
        try:
            if openrouter_api_key:
                print(f"Using OpenRouter with model: {model_name}")
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a Python code generator. Respond only with executable Python code, no explanations or comments except for required pip installations at the top. Return the code within ```python and ``` strings. The first line should be commented out pip install statement"},
                        {"role": "user", "content": f"Generate Python code to {prompt}. If you need to use any external libraries, include a comment at the top of the code listing the required pip installations."}
                    ],
                    max_tokens=4000, temperature=0.7)
                response_content = response.choices[0].message.content
            else: # Fall back to default OpenAI client
                print("OpenRouter API key not found, using default OpenAI client with gpt-4o")
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a Python code generator. Respond only with executable Python code, no explanations or comments except for required pip installations at the top. Return the code within ```python and ``` strings. The first line should be commented out pip install statement"},
                        {"role": "user", "content": f"Generate Python code to {prompt}. If you need to use any external libraries, include a comment at the top of the code listing the required pip installations."}
                    ],
                    max_tokens=4000, temperature=0.7)
                response_content = response.choices[0].message.content
        except Exception as e:
            print(f"Error with OpenRouter: {e}")
            print("Falling back to default OpenAI client with gpt-4o")
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a Python code generator. Respond only with executable Python code, no explanations or comments except for required pip installations at the top. Return the code within ```python and ``` strings. The first line should be commented out pip install statement"},
                        {"role": "user", "content": f"Generate Python code to {prompt}. If you need to use any external libraries, include a comment at the top of the code listing the required pip installations."}
                    ],
                    max_tokens=4000, temperature=0.7)
                response_content = response.choices[0].message.content
            except Exception as e2:
                return f"Failed to generate code: {e2}", e2
        code, error = self.parse_and_exec_code(response_content)
        return code, error  
    def run(self, prompt: str) -> str:
        print(f"Calling Code Generation Tool with the prompt: {prompt}")
        code, error = self.generate_code(prompt)
        if error:
            return f"Code: {code}\n\nCode execution caused an error: {error}"
        return f"Code: {code}\n\n\nCode Executed Successfully"
