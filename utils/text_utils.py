import re


class ActionParsingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


def parse_state(data):
    # 正则表达式匹配节点信息
    pattern = re.compile(r'\[(\d+)\] (\w+) \'(.*?)\'(?: (.*))?')

    # 提取节点信息并转化为字典
    nodes = dict()
    for line in data.splitlines():
        match = pattern.match(line.strip())
        if match:
            node_id = f"[{match.group(1)}]"
            node_type = match.group(2)
            node_text = match.group(3)
            node_state = match.group(4) if match.group(4) else 'None'
            nodes[node_id] = {"type": node_type, "text": node_text, "state": node_state}
    return nodes


def action_completion(action_str, state):
    
    action_str = action_str.strip()     
    nodes = parse_state(state)              # parse node from previous state
    
    if "click " in action_str:
        match = re.search(r"click ?\[(\d+)\]", action_str)
        if not match:
            raise ActionParsingError(f"Invalid click action {action_str}")
        element_id = match.group(1)
        element_id = '['+element_id+']'
        if element_id not in state:
            print(f"Invalid click action ID in {action_str}")
            return None
        return "```"+action_str+"```" + f", where {element_id} is '{nodes[element_id]['text']}'" 
    
    if "hover " in action_str:
        match = re.search(r"hover ?\[(\d+)\]", action_str)
        if not match:
            raise ActionParsingError(f"Invalid hover action {action_str}")
        element_id = match.group(1)
        element_id = '['+element_id+']'
        if element_id not in state:
            print(f"Invalid hover action ID in {action_str}")
            return None
        return "```"+action_str+"```" + f", where {element_id} is '{nodes[element_id]['text']}'" 
    
    if "type " in action_str:
        match = re.search(
            r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str
        )
        if not match:
            print(f"Invalid type action {action_str}")
            return None
        element_id, text, enter_flag = (
            match.group(1),
            match.group(2),
            match.group(3),
        )
        element_id = '['+element_id+']'
        if enter_flag == "1":
            text += "\n"
        return "```"+action_str+"```" + f", where {element_id} is '{nodes[element_id]['text']}'" 
    
    #! un-finished
    if "press " in action_str:
        match = re.search(r"press ?\[(.+)\]", action_str)
        if not match:
            raise ActionParsingError(f"Invalid press action {action_str}")
        key_comb = match.group(1)
    
    #! un-finished
    if "scroll " in action_str:
        # up or down
        match = re.search(r"scroll ?\[?direction=(up|down)\]?", action_str)
        if not match:
            print(f"Invalid scroll action {action_str}")
        else:
            direction = match.group(1)
            return "scroll "+direction
        
        match = re.search(r"scroll ?\[?(up|down)\]?", action_str)
        if not match:
            print(f"Invalid scroll action {action_str}")
            return action_str
        else:
            direction = match.group(1)
            return "scroll "+direction
        
    if "goto " in action_str:
        match = re.search(r"goto ?\[(.+)\]", action_str)
        if not match:
            raise ActionParsingError(f"Invalid goto action {action_str}")
        url = match.group(1)
        
    if "new_tab " in action_str:
        pass
    
    if "go_back " in action_str:
        pass
    
    if "go_forward " in action_str:
        pass
    
    if "tab_focus " in action_str:
        match = re.search(r"tab_focus ?\[(\d+)\]", action_str)
        if not match:
            raise ActionParsingError(
                f"Invalid tab_focus action {action_str}"
            )
        page_number = int(match.group(1))
    
    if "close_tab " in action_str:
        pass
    
    if "stop " in action_str:  # stop answer
        match = re.search(r"stop ?\[(.+)\]", action_str)
        if not match:  # some tasks don't require an answer
            answer = ""
        else:
            answer = match.group(1)
        return "```"+action_str+"```"
    