from openai import OpenAI
import os
import json
client = OpenAI(
    api_key="",
    base_url=""
)
def generate(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        temperature=0.8,  
        top_p=1,  
        presence_penalty=1,  
        max_tokens=500,  
        messages=[
            {"role": "system",
             "content": "I want you to play the role of a mathematical function generator."},
            {"role": "user", "content": prompt}  
        ],
        response_format={
            'type': 'json_object'
        }
    )
    new_output = response.choices[0].message.content.strip()
    return new_output
def read_prompt_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read().strip()
    return prompt
def extract_json_string(new_output):
    new_output = new_output.replace(r'\(', '').replace(r'\)', '')
    start = new_output.find('{')  
    end = new_output.rfind('}')   
    if start == -1 or end == -1:
        return new_output
    new_output = new_output[start:end + 1]
    count_open = new_output.count('{')
    count_close = new_output.count('}')
    if count_open > count_close:
        new_output += '\n}'
    return new_output
def generate_response_suggestion(prompt): 
        role = "I'd like you to act as a mathematician with an active mind, who can analyze and discover patterns in variables from mathematical formulas, thereby better guiding the generation of diverse equations."
        prompt = f"{role}\n{prompt}"
        json_data = {
            "1": "",
            "2": "",
            "3": "",
            "4": "",
        }
        endstr = "You only need to generate 4 suggestions in the format above. Do not generate anything else! Please ensure that the content returned is in strict JSON format! Output only text in txt format, do not output any other special characters!"
        json_string = json.dumps(json_data, indent=2)
        prompt_with_json = f"{prompt}\n{json_string}\n{endstr}"
        new_output = generate(prompt_with_json)
        return extract_json_string(new_output)
def generate_response_separate(prompt):  
    json_data = {
            "1": "x0 + 1",
            "2": "",
            "3": "",
            "4": "",
            "5": "",
    }
    endstr = "You only need to generate 5 equations in the format above. Do not generate anything else! Please ensure that the generated equations are enclosed in "" . Please ensure that the content returned is in strict JSON format!"
    json_string = json.dumps(json_data, indent=2)
    prompt_with_json = f"{prompt}\n{json_string}\n{endstr}"
    new_output = generate(prompt_with_json)
    return extract_json_string(new_output)