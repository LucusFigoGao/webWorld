# An official implementation of "Web World model helps better GUI policy"
## MCTS

### base.node
```python
self.action = action                # str, execute action generated by the Policy model
self.state = state                  # str, generated by the World model
self.parent = parent                # treeNode
self.numVisits = 0                  # int, visiting frequency
self.V = 0                          # float, value of node, generated by the World model
self.children = {}                  # dict{str: treeNode}
self.depth = depth                  # int, tree depth
self.isFullyExpanded = False        # expanded, whether has childnode
self.isTerminal = False             # value acceptable, whether task finished
```

### expand(2025.01.09)
* function: mcts_task.get_reflection(node.trace): string                             // 反思当前节点向上回溯的轨迹是否完成任务(**涉及调用一次反思LLM**)
* function: get_next_step_expand(node, mcts_task): list[string]                      // 根据当前节点进行扩展(广度为B)
    * function: mcts_task.get_next_action(node.state, node.trace): string            // 根据任务`mcts_task.question`, 当前状态`node.state`获得`action`(**涉及调用一次策略LLM**)
    * function: mcts_task.get_next_state_predict(node.state, node.action): string    // 根据当前状态`node.state`, 当前行动`child_node.action`获得子节点的`state`(**涉及调用一次世界LLM**)
    * function: mcts_task.get_step_value(node.state, node.trace): float	             // 根据执行动作后的轨迹`child_node.trace`和状态`child_node.state`获得节点价值(**涉及调用一次奖励LLM**)

思考：
* 基础版本可能先不考虑`mcts_task.get_reflection`，从功能上和奖励模型对轨迹的打分重合了
* `mcts_task.get_next_action`接受当前节点的状态`node.state`，以及轨迹`node.trace`
    * 考虑选择`raw_action`的拼接作为`node.trace`，这里面涉及了策略模型推理过程的思考，如下面的内容：

`node.trace`: 通过<step></step>将`raw_action`拼接起来
```html
    <step>Let's think step-by-step. This page has a link whose ID is [169], which can be used to find directions between two points. 
        In this case, we want directions from Carnegie Mellon University to the top computer science school in Massachusetts, 
        which is MIT. Clicking on this link will probably take me to a page where I can input these two locations and find the 
        driving distance. In summary, the next action I will perform is ```click [169]```.</step>
    <step>Let\'s think step-by-step. To get the distance from Carnegie Mellon University to the top computer science school in 
        Massachusetts, which is Massachusetts Institute of Technology (MIT), I need to input "Carnegie Mellon University" in the 
        "From" textbox and "Massachusetts Institute of Technology" in the "To" textbox, and then click "Go" button. The IDs of the 
        "From" textbox, "To" textbox, and "Go" button are [567], [569], and [556] respectively. First, I will type "Carnegie Mellon 
        University" in the "From" textbox. In summary, the next action I will perform is ```type [567] [Carnegie Mellon University] [0]```</step>
    <step>Let\'s think step-by-step. The previous action was to type Carnegie Mellon University into the "From" box. Now, we need to 
        type the destination into the "To" box. The top computer science school in Massachusetts is Massachusetts Institute of Technology (MIT). 
        So, I will type "MIT" into the "To" box and submit the form by pressing the "Enter" key afterwards. In summary, the next action I 
        will perform is ```type [569] [MIT] [1]```</step>
```

* `mcts_task.get_next_state_predict`接受当前节点的状态`node.state`和采取的行动`child_node.action`给出`child_node.state`
* `mcts_task.get_step_value` 接受轨迹`child_node.trace`,和状态`child_node.state`, 对扩展的子节点进行打分
* 从完整的逻辑看，不管是预测action还是用action作为输入，都是用LLM自己产生的`raw_action`，因此如果不涉及真实环境操作的话，可以不用提取可执行的`action`


### rollout(2025.01.18)
* function: get_best_child(child_node): node                                            // 选择UCB策略筛选出扩展节点中最优的节点
* function: greedy_policy_rollout(node, mcts_task): list[string]                        // 扩展广度为B，模拟深度为K
    while K, do:
    * function: get_next_step_rollout(node, mcts_task): list[string]                    // 根据当前节点进行扩展(广度为B)(**涉及调用3*KB次反思LLM**)

### API请求prompt收集(2025.01.18)
* function: mcts_task.get_next_action(trace, state)
    * [TREE SEARCH FOR LANGUAGE MODEL AGENTS](./prompt.md#action-navigation)
    * [WebArena](prompt.md#p_cot_id_actree_2s_no_na)

* function: mcts_task.get_next_state_predict
    * [WMA-abstraction](./prompt.md#transition-focused-observation-abstraction)
    * [WMA-our-improvement](./prompt.md#ours)

* function: mcts_task.get_step_value
    * [OS-Genesis](./prompt.md#trajectory-reward-model-prompt)
    * [TREE SEARCH FOR LANGUAGE MODEL AGENTS](./prompt.md#reward)
    * [AGENTTREK](./prompt.md#reward-1)

## MCTS (更新&分析)
### select
* 2025.01.09 完成选择部分，这部分不涉及请求大模型，可以直接根据UCB进行节点的遍历

### expand
* 2025.01.09 扩展阶段暂时不考虑加reflection

### rollout
* 2025.01.18 完成随机/贪婪模拟阶段，同样暂时不考虑加reflection

### back propagation
* 2025.01.18 完成回溯部分，保留了和ReST-MCTS*相同的回溯策略

