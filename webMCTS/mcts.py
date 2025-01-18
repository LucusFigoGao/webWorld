import math
import time
import numpy
import random
import random as rd

from webMCTS.base import treeNode
from webMCTS.task import MCTS_Task

# select
def selectNode(node: treeNode, mcts_task: MCTS_Task):
    while node.isFullyExpanded:
        node = getBestChild(node, mcts_task)
    if isTerminal(node, mcts_task):
        node.isTerminal = True
        return True, node
    else:
        return False, node


def getBestChild(node: treeNode, mcts_task: MCTS_Task):
    """
        UCB(C) = v_{C} + \epsilon * \sqrt{ \frac{\ln{n_{parent}}}{n_{C}} }
        :: Hint: If n_{C} is 0, it may lead to the stackoverlow, 
        :: Hint: thus set mcts_task.INF = 1 to avoid this happen.
        :: Hint: If len(bestNodes) > 1, which means more than one node meet
        :: Hint: the UCB criteria, thus randomly select one node.
    """
    best_UCB_value = 0
    bestNodes: list = []
    
    for child in node.children.values():
        child: treeNode
        child_UCB_value = child.V + mcts_task.exploration_constant * math.sqrt(
            2 * math.log(node.numVisits) / child.numVisits
        ) if child.numVisits > 0 else child.V + mcts_task.INF
        
        if child_UCB_value >= best_UCB_value:
            best_UCB_value = child_UCB_value
            bestNodes.append(child)
            
    return rd.choice(bestNodes)


def isTerminal(node: treeNode, mcts_task):
    if mcts_task.reward_model_type == 'vm':
        return node.V >= mcts_task.end_gate
    else:
        return False

# expand
def get_next_step_expand(node: treeNode, mcts_task: MCTS_Task):
    action_list = []
    
    for i in range(mcts_task.branch):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            proposal = mcts_task.get_next_action(trace=node.trace, state=node.state)
            cnt -= 1
        if not proposal:
            continue
        action_list.append(proposal)
    
    for action in action_list:
        if action not in node.children.keys():
            
            node.append_children(action)
            
            child: treeNode = node.children[action]
                
            child.update_state(mcts_task.get_next_state_predict(state=node.state, action=action))
            
            child.update_value(mcts_task.get_step_value(child.trace, child.state))
            
    node.isFullyExpanded = True
    
    return node


def expand(node: treeNode, mcts_task: MCTS_Task):
    """
        :: 这里分两步，预留出reflection的接口，用于后续加reflection；下一步是`get_next_step_expand`
    """
    # step1
    
    # step two
    node = get_next_step_expand(node, mcts_task)
    pass

# rollout
def get_next_step_random_rollout(trace, state, mcts_task: MCTS_Task):
    # get next action
    action_list = []
    for i in range(mcts_task.roll_branch):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            proposal = mcts_task.get_next_action(trace=trace, state=state)
            cnt -= 1
        if not proposal:
            continue
        action_list.append(proposal)
    
    action = random.choice(action_list)
    new_trace = trace + action
    new_state = mcts_task.get_next_state_predict(state=state, action=action)
    new_value = mcts_task.get_step_value(new_trace, new_state)
    return new_trace, new_state, new_value
    

def randomPolicy(node: treeNode, mcts_task: MCTS_Task):
    max_V = mcts_task.low
    trace = node.trace
    state = node.state
    cur_step = node.depth + 1
    
    for i in range(mcts_task.roll_forward_steps):
        trace, state, value = get_next_step_random_rollout(trace, state, mcts_task)
        cur_step += 1
        if value > max_V:
            max_V = value
        
    return max_V


def get_next_step_greedy_rollout(trace, state, mcts_task: MCTS_Task):
    # get next action
    action_list = []
    for i in range(mcts_task.roll_branch):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            proposal = mcts_task.get_next_action(trace=trace, state=state)
            cnt -= 1
        if not proposal:
            continue
        action_list.append(proposal)
    new_traces = [trace + action for action in action_list]
    
    # get next state predict
    new_states = [mcts_task.get_next_state_predict(state=state, action=action) for action in action_list]
    
    # get step value
    new_values = [mcts_task.get_step_value(t, s) for t, s in zip(new_traces, new_states)]
    
    return new_traces, new_states, new_values


def greedyPolicy(node: treeNode, mcts_task: MCTS_Task):
    max_V = mcts_task.low
    trace = node.trace
    state = node.state
    cur_step = node.depth + 1
    
    for i in range(mcts_task.roll_forward_steps):
        new_traces, new_states, new_values = get_next_step_greedy_rollout(trace, state)
        cur_step += 1
        idx = numpy.argmax(new_values)
        trace, state, value = new_traces[idx], \
                                             new_states[idx], \
                                             new_values[idx]
        if value > max_V:
            max_V = value
    
    return max_V
        
# back propagate
def back_propagate(node: treeNode):
    while node is not None:
        node.numVisits += 1
        if node.isFullyExpanded:
            child_Vs = [child.V * child.numVisits for child in node.children.values()]
            total_num_visits = sum([child.numVisits for child in node.children.values()])
            if total_num_visits > 0:
                node.V = sum(child_Vs) / total_num_visits
        node = node.parent


def executeRound(root: treeNode, mcts_task: MCTS_Task):
    
    print('-' * 40, '\n选择节点阶段\n')
    flag, node = selectNode(root, mcts_task)
    
    if flag:  # task finished
        return True, node, root
    
    print('-' * 40, '\n扩充阶段\n')
    node = expand(node, mcts_task)
    
    print('-' * 40, '\n模拟搜索阶段\n')
    roll_node = getBestChild(node, mcts_task)
    best_V = greedyPolicy(roll_node, mcts_task) if mcts_task.roll_policy == 'greedy' else randomPolicy(roll_node, mcts_task)
    roll_node.V = roll_node.V * (1 - mcts_task.alpha) + best_V * mcts_task.alpha
    roll_node.numVisits += 1
    
    print('-' * 40, '\n反向传播阶段\n')
    back_propagate(node)
    
    return False, node, root
    

def MCTS_search(mcts_task: MCTS_Task):
    root = treeNode(action='')
    root.update_state(state=mcts_task.init_state)   # update the initial state

    if mcts_task.limit_type == 'time':
        timeLimit = time.time() + mcts_task.time_limit / 1000
        time_start = time.time()
        while time.time() < timeLimit:
            print(f'<开始新搜索轮次，目前总时间:{time.time() - time_start}>\n')
            flag, node, root = executeRound(root, mcts_task)
            if flag:
                print('已找到解决方案！\n')
                return root, node, time.time() - time_start
    else:
        for i in range(mcts_task.iteration_limit):
            print(f'<开始新搜索轮次，目前已完成轮次数:{i}>\n')
            flag, node, root = executeRound(root, mcts_task)
            if flag:
                print('已找到解决方案！\n')
                return root, node, i + 1
    return root, None, None


def MCTS(mcts_task: MCTS_Task):
    root, node, finish = MCTS_search(mcts_task)
    pass
