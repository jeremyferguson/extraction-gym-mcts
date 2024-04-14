use super::*;
use std::collections::{BinaryHeap, HashSet};
use std::f64::consts::SQRT_2;

#[derive(PartialEq, Eq, Hash)]
struct MCTSChoice {
    class: ClassId,
    node: NodeId,
}
struct MCTSNode {
    to_visit: HashSet<ClassId>,
    decided_classes: FxHashMap<ClassId, NodeId>,
    num_rollouts: i32,
    min_cost: f64,
    min_cost_map: FxHashMap<ClassId, NodeId>,
    edges: FxHashMap<MCTSChoice, MCTSNode>,
    parent: Option<Box<MCTSNode>>,
    parent_edge: Option<MCTSChoice>,
    explored: bool,
}

const EXPLORATION_PARAM: f64 = SQRT_2;

pub struct MCTSExtractor;
impl MCTSExtractor {
    fn mcts(&self, egraph: &EGraph, root: ClassId, num_iters: i32) -> FxHashMap<ClassId, NodeId> {
        let mut root_node = MCTSNode {
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
                egraph.classes().len(), //TODO: check if this is correct
                Default::default(),
            ),
            parent: None,
            parent_edge: None,
            explored: false,
        };
        for _ in 0..num_iters {
            let leaf = self.choose_leaf(&root_node, egraph);
            match leaf {
                Some(node) => {
                    let (choices, new_node) = self.rollout(node, egraph);
                    self.backprop(new_node, choices);
                }
                None => break,
            };
        }
        return root_node.min_cost_map;
    }
    fn choose_leaf<'a>(&'a self, root: &'a MCTSNode, egraph: &EGraph) -> Option<&MCTSNode> {
        let mut curr = root;
        loop {
            // look for a choice not in curr's edges
            for class_id in curr.to_visit.iter() {
                for node_id in egraph.classes().get(class_id).unwrap().nodes.iter() {
                    if !curr.edges.contains_key(&MCTSChoice { class: class_id.clone(), node: (*node_id).clone() }) {
                        return Some(curr);
                    }
                }
            }
            // if we get here, then all choices are in curr's edges
            // filter edges for ones that are not explored
            let unexplored_children: Vec<&MCTSNode> = curr.edges.values().filter(|n| !n.explored).collect();
            // if there are no unexplored children, then we've completely explored the tree
            if unexplored_children.is_empty() { return None; }
            // map nodes to uct cost and choose node which maximizes uct
            curr = self.uct_choose_from_nodes(unexplored_children, curr.num_rollouts);
        }
    }
    fn uct_choose_from_nodes<'a>(&'a self, nodes: Vec<&'a MCTSNode>, parent_num_rollouts: i32) -> &MCTSNode {
        // pre-compute min and max cost
        let mut min_cost: f64 = f64::INFINITY;
        let mut max_cost: f64 = f64::NEG_INFINITY;
        for node in nodes.iter() {
            min_cost = min_cost.min(node.min_cost);
            max_cost = max_cost.max(node.min_cost)
        }
        // Initialize an empty max heap
        let mut uct_heap: BinaryHeap<(NotNan<f64>, usize)> = BinaryHeap::with_capacity(nodes.len());
        // For each node, compute uct and insert into heap
        for (i, node) in nodes.iter().enumerate() {
            let uct_cost = self.compute_uct(node.min_cost, min_cost, max_cost, node.num_rollouts as f64, parent_num_rollouts as f64);
            uct_heap.push((NotNan::new(uct_cost).unwrap(), i))
        }
        let (_, node_index) = uct_heap.pop().unwrap();
        nodes[node_index]
    }
    fn compute_uct(&self, cost: f64, min_cost: f64, max_cost: f64, num_rollouts: f64, parent_num_rollouts: f64) -> f64 {
        let cost_term = 1.0 - ((cost - min_cost) / (max_cost - min_cost));
        let rollout_term = EXPLORATION_PARAM * (parent_num_rollouts.ln() / num_rollouts).sqrt();
        cost_term + rollout_term
    }
    fn rollout(&self, node: &MCTSNode, egraph: &EGraph) -> (FxHashMap<ClassId, NodeId>, &MCTSNode) {
        return (
            FxHashMap::<ClassId, NodeId>::with_capacity_and_hasher(
                egraph.classes().len(),
                Default::default(),
            ),
            node,
        );
        //TODO: Jeremy
    }
    fn backprop(&self, new_node: &MCTSNode, choices: FxHashMap<ClassId, NodeId>) -> () {
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
