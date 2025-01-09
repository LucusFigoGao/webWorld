import time

from webMCTS.base import treeNode


def selectNode(node, mcts_task):
    while node.isFullyExpanded:
        node = getBestChild(node, mcts_task)
    if isTerminal(node, mcts_task):
        node.final_ans_flag = 1
        return True, node
    else:
        return False, node


def getBestChild(node, mcts_task):
    pass


def isTerminal(node, mcts_task):
    pass


def expand(node, mcts_task):
    pass
    
    
def greedyPolicy(node, mcts_task):
    pass


def randomPolicy(node, mcts_task):
    pass


def back_propagate(node):
    pass
    

def executeRound(root, mcts_task):
    
    print('-' * 40, '\n选择节点阶段\n')
    flag, node = selectNode(root, mcts_task)
    
    
    print('-' * 40, '\n扩充阶段\n')
    node = expand(node, mcts_task)
    
    print('-' * 40, '\n模拟搜索阶段\n')
    roll_node = getBestChild(node, mcts_task)
    best_V = greedyPolicy(roll_node, mcts_task) if mcts_task.roll_policy == 'greedy' else randomPolicy(roll_node, mcts_task)
    
    print('-' * 40, '\n反向传播阶段\n')
    back_propagate(node)
    
    return False, node, root
    


def MCTS_search(mcts_task):
    root = treeNode(action='', state=mcts_task.init_state)

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


def MCTS(mcts_task):
    root, node, finish = MCTS_search(mcts_task)
    pass
