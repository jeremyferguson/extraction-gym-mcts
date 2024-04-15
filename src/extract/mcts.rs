use super::*;
use std::collections::{BinaryHeap, HashSet};
use std::f64::consts::SQRT_2;
use rand::seq::SliceRandom;
use std::rc::Rc;
use std::rc::Weak;

#[derive(PartialEq, Eq, Hash,Clone)]
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
    edges: FxHashMap<MCTSChoice, Box<MCTSNode>>,
    parent: Option<Box<MCTSNode>>,
    parent_edge: Option<MCTSChoice>,
    explored: bool,
}

const EXPLORATION_PARAM: f64 = SQRT_2;

pub struct MCTSExtractor;
impl MCTSExtractor {
    fn mcts<'b>(&mut self, egraph: &EGraph, root: ClassId, num_iters: i32) -> FxHashMap<ClassId, NodeId> {
        let root_node = Box::new(MCTSNode {
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
            edges: FxHashMap::<MCTSChoice, Box<MCTSNode>>::with_capacity_and_hasher(
                egraph.classes().len(), //TODO: check if this is correct
                Default::default(),
            ),
            parent: None,
            parent_edge: None,
            explored: false,
        });
        for _ in 0..num_iters {
            let leaf = self.choose_leaf(root_node, egraph);
            match leaf {
                Some(mut node) => {
                    let (choices, new_node) = self.rollout(node, egraph);
                    self.backprop(&new_node, &choices);
                }
                None => break,
            };
        }
        return root_node.min_cost_map;
    }
    fn choose_leaf(&self, root: Box<MCTSNode>, egraph: &EGraph) -> Option<Box<MCTSNode>> {
        let mut curr = root;
        loop {
            // look for a choice not in curr's edges
            for class_id in curr.to_visit.iter() {
                for node_id in egraph.classes().get(class_id).unwrap().nodes.iter() {
                    if !curr.edges.contains_key(&MCTSChoice { class: (*class_id).clone(), node: (*node_id).clone() }) {
                        return Some(curr);
                    }
                }
            }
            // if we get here, then all choices are in curr's edges
            // filter edges for ones that are not explored
            let unexplored_children: Vec<Box<MCTSNode>> = curr.edges.values().filter(|n| !n.explored).map(|n| *n).collect();
            // if there are no unexplored children, then we've completely explored the tree
            if unexplored_children.is_empty() { return None; }
            // map nodes to uct cost and choose node which maximizes uct
            curr = self.uct_choose_from_nodes(unexplored_children, curr.num_rollouts);
        }
    }
    fn uct_choose_from_nodes(& self, nodes: Vec<Box<MCTSNode>>, parent_num_rollouts: i32) -> Box<MCTSNode> {
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
    fn find_first_node(& self, node: Box<MCTSNode>, egraph: & EGraph) -> Option<MCTSChoice> {
        for class_id in node.to_visit.iter(){
            for node_id in egraph.classes().get(class_id).unwrap().nodes.iter(){
                if !node.edges.contains_key(&MCTSChoice { class: (*class_id).clone(), node: (*node_id).clone() }){
                    return Some(MCTSChoice { class: (*class_id).clone(), node: (*node_id).clone() });
                }
            }
        }
        return None;
    }
    fn rollout(&self, node: Box<MCTSNode>, egraph: &EGraph) -> ( Box<FxHashMap<ClassId, NodeId>>, Box<MCTSNode>) {
        let first_choice : MCTSChoice = self.find_first_node(node,egraph).unwrap();
        let choices : Box<FxHashMap<ClassId, NodeId>> = Box::new(node.decided_classes.clone());
        let mut new_to_visit = node.to_visit.clone();

        let mut new_decided : FxHashMap<ClassId, NodeId> = *choices.clone();
        new_decided.insert(first_choice.class.clone(),first_choice.node.clone());
        new_to_visit.remove(&first_choice.class);

        let new_node = MCTSNode{
            to_visit: new_to_visit,
            decided_classes: new_decided,
            num_rollouts: 0,
            min_cost: f64::INFINITY,
            min_cost_map: FxHashMap::<ClassId, NodeId>::with_capacity_and_hasher(
                egraph.classes().len(),
                Default::default(),
            ),
            edges: FxHashMap::<MCTSChoice, Box<MCTSNode>>::with_capacity_and_hasher(
                egraph.classes().len(), //TODO: check if this is correct
                Default::default(),
            ),
            parent: Some(node),
            parent_edge: Some(first_choice.clone()),
            explored: false
        };
        let new_node = Box::new(new_node);
        node.edges.insert(first_choice.clone(),new_node); 
        let mut todo = new_to_visit.clone();
        //First iteration of the loop: remove first choice from todo list
        //add children to todo list
        todo.remove(&first_choice.class);
        choices.insert(first_choice.class.clone(),first_choice.node.clone());
        let children = egraph[&first_choice.node].children.clone().into_iter().map(|n| egraph.nid_to_cid(&n)).filter(
            |n| !choices.contains_key(n));
        for child in children{
            todo.insert(child.clone());
        }
        while !todo.is_empty(){
            //randomly choose a class from todo, and a node from the class
            let to_visit_vec : Vec<ClassId> = todo.clone().into_iter().collect();
            let class_choice = to_visit_vec.choose(&mut rand::thread_rng()).unwrap();
            let node_choice = egraph[class_choice].nodes.choose(&mut rand::thread_rng()).unwrap();
            let choice = MCTSChoice { class: (*class_choice).clone(), node: (*node_choice).clone() };
            //add choice to choices and add children of choice to todo
            choices.insert(choice.class.clone(),choice.node.clone());
            let children = egraph[&choice.node].children.clone().into_iter().map(|n| egraph.nid_to_cid(&n)).filter(
                |n| !choices.contains_key(n));
            for child in children{
                todo.insert(child.clone());
            }
            todo.remove(&choice.class);
        }
        return (choices,new_node);
    }
    fn backprop(&self, _new_node: &MCTSNode, _choices: &FxHashMap<ClassId, NodeId>) -> () {
        //TODO: Jacob
    }
}
impl Extractor for MCTSExtractor {
    fn extract(&self, _egraph: &EGraph, _roots: &[ClassId]) -> ExtractionResult {
        //TODO: Jacob
        let result = ExtractionResult::default();
        return result;
    }
}
