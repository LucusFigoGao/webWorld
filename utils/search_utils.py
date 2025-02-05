import json


def get_all_leaf_nodes(root):
    """深度优先搜索（DFS）"""
    leaves = []
    if not root:
        return leaves
    stack = [root]
    while stack:
        current_node = stack.pop()
        if not current_node.children:
            leaves.append(current_node)
        else:
            stack.extend(current_node.children.values())
    return leaves


def save_tree(root, filename):
    """将树保存为 JSON 文件"""
    tree_dict = serialize_node(root)
    with open(filename, 'w') as f:
        json.dump(tree_dict, f, indent=2)


def serialize_node(node):
    """递归序列化单个节点及其子节点"""
    if not node:
        return None
    return {
        "action": node.action,
        "state": node.state,
        "numVisits": node.numVisits,
        "V": node.V,
        "depth": node.depth,
        "isTerminal": node.isTerminal,
        "children": {action: serialize_node(child) for action, child in node.children.items()}
    }