import os 
import json 
from dotenv import load_dotenv
import google.generativeai as genai 
import prompts



def setup_api_key():
    
    load_dotenv()
    num_keys = 11
    
    with open('key_rotation.json', 'r') as f:
        rot_config = json.load(f)
    curr_rotation = rot_config['CURRENT_ROTATION']
    
    key_name = "GOOGLE_API_KEY_" + str(curr_rotation)
    genai.configure(api_key=os.getenv(key_name))

    next_rotation = (curr_rotation+1)%num_keys
    rot_config['CURRENT_ROTATION'] = next_rotation
    with open('key_rotation.json', 'w') as f:
        json.dump(rot_config, f, indent=4)
        
        


# Function to get the response from gemini-1.0-pro
def get_gemini_10_response(question:str, prompt:str):
    setup_api_key()
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt, question])
    return response.text

# Function to get the response from gemini-1.5-pro
def get_gemini_15_response(question:str, prompt:str):
    setup_api_key()
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([prompt, question])
    return response.text





# Function to generate the field values from the input question
def generate_field_values(question:str):
    prompt = prompts.analysis_prompt()
    response = get_gemini_10_response(question=question, prompt=prompt)
    return response  

#Function to generate the general response
def generate_general_output(question:str):
    prompt = prompts.general_prompt()
    response = get_gemini_10_response(question=question, prompt=prompt)
    return response

# Function to generate image response
def generate_image_output(question:str):
    prompt = prompts.image_prompt()
    response = get_gemini_10_response(question=question, prompt=prompt)
    return response

# Function to generate plot response
def generate_plot_output(question:str):
    prompt = prompts.plot_prompt()
    response = get_gemini_10_response(question=question, prompt=prompt)
    return response