import json

def load_json(file_name: str):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data