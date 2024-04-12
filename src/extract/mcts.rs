use super::*;

struct MCTSChoice{
    class:ClassId,
    node:NodeId
}
struct MCTSNode{
    to_visit: FxHashSet<ClassId>,
    decided_classes: FxHashMap<ClassId,NodeId>,
    num_rollouts: i32,
    min_cost: f64,
    min_cost_map: FxHashMap<ClassId,NodeId>,
    edges: FxHashMap<MCTSChoice,MCTSNode>,
    parent: Option<MCTSNode>,
    parent_edge: Option<MCTSChoice>,
    explored: bool
}

pub struct MCTSExtractor;
impl Extractor for MCTSExtractor {
    fn extract(&self, egraph: &EGraph, _roots: &[ClassId]) -> ExtractionResult {
        //TODO: fill in
    }
    fn MCTS(&self, egraph: &EGraph, root: ClassId,num_iters: i32) -> FxHashMap<ClassId,NodeId> {
        let mut root_node = MCTSNode{
            to_visit: FxHashSet::from([root]);,
            decided_classes: FxHashMap::new(),
            num_rollouts: 0,
            min_cost: INFINITY,
            min_cost_map: FxHashMap::new(),
            edges: FxHashMap::new(),
            parent: None,
            parent_edge: None,
            explored: false
        };
        for _ in 0..num_iters {
            let leaf = chooseLeaf(root_node);
            if leaf == None{
                break;
            }
            let (choices,new_node) = rollout(leaf,egraph);
            backprop(new_node,choices);
        }
        return root_node.min_cost_map;
    }
    fn chooseLeaf(root: MCTSNode) -> Option<MCTSNode>{
        //TODO: fill in
    }
    fn rollout(node:MCTSNode,egraph:EGraph) -> (FxHashMap<ClassId,NodeId>, MCTSNode) {
        //TODO: fill in 
    }
    fn backprop(new_node:MCTSNode,choices:FxHashMap<ClassId,NodeId>) -> (){
        //TODO: fill in
    }
}
