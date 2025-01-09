import math
import time
import random as rd

from webMCTS.base import treeNode
from webMCTS.task import MCTS_Task


def selectNode(node: treeNode, mcts_task: MCTS_Task):
    while node.isFullyExpanded:
        node = getBestChild(node, mcts_task)
    if isTerminal(node, mcts_task):
        node.isTerminal = True
        return True, node
    else:
        return False, node

# 根据UCB原则选择最优的节点
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


def expand(node: treeNode, mcts_task: MCTS_Task):
    """
        :: 这里分两步，预留出reflection的接口，用于后续加reflection；下一步是`get_next_step_expand`
    """
    # step1
    
    # step two
    node = get_next_step_expand(node, mcts_task)
    pass


def get_next_step_expand(node: treeNode, mcts_task: MCTS_Task):
    action_list = []
    
    for i in range(mcts_task.branch):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            proposal = mcts_task.get_next_step(trace=node.trace, state=node.state)
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


def greedyPolicy(node: treeNode, mcts_task: MCTS_Task):
    pass


def randomPolicy(node: treeNode, mcts_task: MCTS_Task):
    pass


def back_propagate(node: treeNode):
    pass
    

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
