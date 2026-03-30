import json
import torch
import utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_args = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 100,
        "num_beams": 1,
        "max_length": 1024,
        "min_length": 0,
        "tokenizer_pad": "\\[PAD\\]",
        "tokenizer_padding_side": "left",
        "seed": -1,
        "api_key_path": None,
        "organization_id_path":None,
    }
dtype = torch.float16
model_1 = utils.load_model("/model",device, dtype, "/model", model_args)
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
        }
        endstr = "You only need to generate 3 suggestions in the format above. Do not generate anything else! Please ensure that the content returned is in strict JSON format! Output only text in txt format, do not output any other special characters!"
        json_string = json.dumps(json_data, indent=2)
        prompt_with_json = f"{prompt}\n{json_string}\n{endstr}"
        new_output = model_1.generate(prompt_with_json, return_prompt=False, temperature=0.7)
        return extract_json_string(new_output)

def generate_response_separate(prompt): 
    json_data = {
            "1": "(x0+x1)/(x1)",
            "2": "",
            "3": "",
            "4": "",
            "5": "",
    }
    endstr = "You only need to generate 5 equations in the format above. Do not generate anything else! Please ensure that the generated equations are enclosed in "" . Please ensure that the content returned is in strict JSON format!"
    json_string = json.dumps(json_data, indent=2)
    prompt_with_json = f"{prompt}\n{json_string}\n{endstr}"
    new_output = model_1.generate(prompt_with_json, return_prompt=False, temperature=0.7)
    return extract_json_string(new_output)