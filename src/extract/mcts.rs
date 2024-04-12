use super::*;
use std::collections::HashSet;
struct MCTSChoice{
    class:ClassId,
    node:NodeId
}
struct MCTSNode{
    to_visit: HashSet<ClassId>,
    decided_classes: FxHashMap<ClassId,NodeId>,
    num_rollouts: i32,
    min_cost: f64,
    min_cost_map: FxHashMap<ClassId,NodeId>,
    edges: FxHashMap<MCTSChoice,MCTSNode>,
    parent: Option<Box<MCTSNode>>,
    parent_edge: Option<MCTSChoice>,
    explored: bool
}

pub struct MCTSExtractor;
impl MCTSExtractor{
    fn mcts(&self, egraph: &EGraph, root: ClassId,num_iters: i32) -> FxHashMap<ClassId,NodeId> {
        let mut root_node = MCTSNode{
            to_visit: HashSet::from([root]),
            decided_classes: FxHashMap::<ClassId, NodeId>::with_capacity_and_hasher(
                egraph.classes().len(),
                Default::default(),
            ),
            num_rollouts: 0,
            min_cost: f64::INFINITY,
            min_cost_map: FxHashMap::<ClassId, NodeId>::with_capacity_and_hasher(
                egraph.classes().len(),
                Default::default(),
            ),
            edges: FxHashMap::<MCTSChoice, MCTSNode>::with_capacity_and_hasher(
                egraph.classes().len(),//TODO: check if this is correct
                Default::default(),
            ),
            parent: None,
            parent_edge: None,
            explored: false
        };
        for _ in 0..num_iters {
            let leaf = self.choose_leaf(&root_node);
            match leaf {
                Some(node) =>{
                    let (choices,new_node) = self.rollout(node,egraph);
                    self.backprop(new_node,choices);},
                None => break
            };
        }
        return root_node.min_cost_map;
    }
    fn choose_leaf(&self,root: &MCTSNode) -> Option<MCTSNode>{
        //TODO: Sora
        return None;
    }
    fn uct(&self)->(){
        //TODO: Sora
    }
    fn rollout(&self,node:MCTSNode,egraph:&EGraph) -> (FxHashMap<ClassId,NodeId>, MCTSNode) {
        return (FxHashMap::<ClassId, NodeId>::with_capacity_and_hasher(
            egraph.classes().len(),
            Default::default(),
        ),node);
        //TODO: Jeremy
    }
    fn backprop(&self,new_node:MCTSNode,choices:FxHashMap<ClassId,NodeId>) -> (){
        //TODO: Jacob
    }
}
impl Extractor for MCTSExtractor {
    fn extract(&self, egraph: &EGraph, _roots: &[ClassId]) -> ExtractionResult {
        //TODO: Jacob
        let mut result = ExtractionResult::default();
        return result;
    }
    
}
