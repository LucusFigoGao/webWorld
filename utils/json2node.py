from webMCTS.base import treeNode


def build_tree_from_json(json_data, parent=None, depth=0):
    # 创建当前节点
    node = treeNode(action=json_data.get('action', ''), parent=parent, depth=depth)
    # 设置节点属性
    node.state = json_data.get('state', '')
    node.numVisits = json_data.get('numVisits', 0)
    node.V = json_data.get('V', 0.0)
    node.isTerminal = json_data.get('isTerminal', False)
    # 递归构建子节点
    children_data = json_data.get('children', {})
    # 判断节点是否扩展
    node.isFullyExpanded = True if len(list(children_data.keys())) > 0 else False
    node.update_trace_from_parent()

    for action_key, child_data in children_data.items():
        child_node = build_tree_from_json(child_data, parent=node, depth=depth+1)
        node.children[action_key] = child_node
    return node