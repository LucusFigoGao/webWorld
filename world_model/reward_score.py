from openai import OpenAI
import json
from bs4 import BeautifulSoup
import time

def get_response(prompt: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content


def process_html(html_path: str):
    def get_trace(raw_prediction: list):
        trace = []
        for i in range(len(raw_prediction)):
            text = raw_prediction[i].text
            text = text.split("<pre>")[-1].split("</div>")[0].replace("</pre>", "</step>\n")
            text = "<step>" + text + "</step>\n"
            trace.append(text)
        trace_text = ''.join(trace)
        return trace_text

    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    # 从html中获取数据
    soup = BeautifulSoup(html, 'html.parser')
    observations = soup.find_all("div", {"class": "state_obv"})
    urls = soup.find_all("h3", {"class": "url"})
    raw_predictions = soup.find_all("div", {"class": "raw_parsed_prediction"})
    actions = soup.find_all("div", {"class": "action_object"})

    states = []
    state_obv_div = observations[0].find_all("pre")
    for i in state_obv_div:
        if i.text.startswith("Tab 0 (current):"):
            states.append(i.text)
    indent = soup.text.split("intent:")[-1].split("require_reset")[0].strip()
    trace = get_trace(raw_predictions)
    # state_obv_div[0].find_all("pre")[4].text

    return states, trace, indent


def eval_all_actions(state: str, trace: str, indent: str):
    prompt = f'''You are an expert in evaluating GUI agent task trajectories. Your task is to assess the quality and effectiveness of task trajectories for GUI manipulation tasks.
A trajectory consists of the following components:
1. High-level Instruction: Describes the user’s intended task.
2. Action History: Includes two key parts:
- Reasoning and Action for Each Step: A sequence of actions performed by the agent, including the reasoning thought and final executed action.
- The current web page's accessibility tree of the last state: This is a simplified representation of the webpage, providing key information.
When evaluating a trajectory, consider these key aspects:
Evaluation Criteria:
1. Trajectory Coherence:
- Do the low-level steps and corresponding actions follow a logical sequence toward the goal?
- Are the actions clearly described and specific?
- Are there redundant or unnecessary actions?
2. Task Completion:
- Does the trajectory successfully achieve the instructed task?
- Are all necessary interactions completed?
- Are error cases handled appropriately?
Scoring Guidelines:
Rate the trajectory on a scale of 1 to 5 based on the evaluation criteria:
- 5: The task is perfectly completed, successfully executing multiple actions to achieve the goal. The sequence is logically clear with no noticeable redundancies.
- 4: The task is mostly completed, successfully executing multiple actions. However, due to challenges or ambiguities in the instructions, the completion is not perfect, or there are inefficiencies in the process.
- 3: The task is partially completed, with some successful actions executed. However, due to task or environmental constraints, the goal is not fully achieved, or the sequence ends in a loop or error.
- 2: Only a few actions are executed. Although there is an attempt to complete the task, the trajectory deviates from the goal early on or demonstrates significant inefficiencies in execution and logic.
- 1: The task fails completely, with no meaningful actions executed at the start. The sequence either falls into an immediate deadlock, a repetitive loop, or demonstrates no value in completing the task.
Or the tasks are completely inaccessible.
Note: If the task is relatively complex, but the trajectory demonstrates valuable attempts, even if the task is not fully completed, consider adjusting the score upward. However, if the task is complex but the trajectory fails to perform actions that contribute meaningfully to task completion, no extra points should be awarded.
You need to judge the score based on the agent’s actions and screenshots combined.
Response Format:
Format your response into two lines as shown below:
Reason: <your thoughts and reasoning process for the score>
Score: <your score from 1-5>

** High-level Instruction **:{indent}
** Action History **:
- Reasoning and Action for Each Step:
{trace}
The current web page's accessibility tree of the last state:
{state}

** Your Response **:
'''
    response = get_response(prompt)
    return response


if __name__ == "__main__":
    client = OpenAI(api_key="sk-9ea2917d5f144d3289368cd040d67a85", base_url="https://api.deepseek.com")
    result = []
    for i in range(1, 20):
        state, trace, indent = process_html("dataset/webarena/reasoning/render_{}.html".format(i))
        current_state = state[-1]
        response = eval_all_actions(current_state, trace, indent)
        print("-" * 20, i, "-" * 20)
        print(response)
        result.append(response)
        with open("output/reward_all_action.json", "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        time.sleep(1)
