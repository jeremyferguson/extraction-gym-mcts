use super::*;
use std::collections::{BinaryHeap, HashSet};
use std::f64::consts::SQRT_2;
use rand::seq::SliceRandom;

#[derive(PartialEq, Eq, Hash,Clone)]
struct MCTSChoice {
    class: ClassId,
    node: NodeId,
}
#[derive(Clone)]
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
const NUM_ITERS: i32 = 100;

pub struct MCTSExtractor;
impl MCTSExtractor {
    //helper function
    fn compute_mcts_tree_size(&self, root: &Box<MCTSNode>) -> i32{
        let mut sum: i32 = 1;
        for (_, node) in root.edges.iter(){
            sum += self.compute_mcts_tree_size(node);
        }
        return sum;
    }

    fn mcts(&self, egraph: &EGraph, root: ClassId, num_iters: i32) -> FxHashMap<ClassId, NodeId> {
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
            let leaf = self.choose_leaf(root_node.clone(), egraph);
            match leaf {
                Some(mut node) => {
                    //let (choices, new_node) = self.rollout(node, egraph);
                    //println!("Tree size: {}",self.compute_mcts_tree_size(&root_node.clone()));
                    //self.backprop(egraph, new_node, choices);
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
            let unexplored_children: Vec<Box<MCTSNode>> = curr.edges.values().filter(|n| !n.explored).map(|n| Box::new(*(n.clone()))).collect();
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
        nodes[node_index].clone()
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
    fn rollout(&self, mut node: Box<MCTSNode>, egraph: &EGraph) -> ( Box<FxHashMap<ClassId, NodeId>>, Box<MCTSNode>) {
        // get an MCTSChoice to take from the leaf node
        let first_choice : MCTSChoice = self.find_first_node(node.clone(),egraph).unwrap();

        // initialize the new MCTSNode
        // initialize the MCTSNode's decided map and add the MCTSChoice we chose above
        let mut new_decided : FxHashMap<ClassId, NodeId> = node.decided_classes.clone();
        new_decided.insert(first_choice.class.clone(),first_choice.node.clone());
        // initialize a set of e-classes we need to visit
        let mut new_to_visit = node.to_visit.clone();
        // remove the e-class corresponding to the choice made above
        new_to_visit.remove(&first_choice.class);
        // insert the e-classes of the children of the node we chose above
        let children = egraph[&first_choice.node].children.clone().into_iter()
            .map(|n| egraph.nid_to_cid(&n))
            .filter(|n| !new_decided.contains_key(n));
        for child in children{
            new_to_visit.insert(child.clone());
        }
        let new_node = Box::new(MCTSNode{
            to_visit: new_to_visit.clone(),
            decided_classes: new_decided.clone(),
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
            parent: Some(node.clone()),
            parent_edge: Some(first_choice.clone()),
            explored: false
        });
        // update the leaf node to have an edge pointing to the node we just created
        node.edges.insert(first_choice.clone(), new_node.clone());

        // clone new_to_visit to have a todo list for our rollout
        let mut todo = new_to_visit.clone();
        // initialize a map of choices taken on this rollout
        let mut choices : Box<FxHashMap<ClassId, NodeId>> = Box::new(new_decided.clone());

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
        return (choices, new_node);
    }


    fn backprop(&self, egraph: &EGraph, mut current: Box<MCTSNode>, mut choices: Box<FxHashMap<ClassId, NodeId>>) -> () {
        loop {
            // compute cost
            let choices_cost = self.cost(egraph, choices.clone());
            // update choices & cost of current node
            if choices_cost < current.min_cost {
                current.min_cost = choices_cost;
                current.min_cost_map = (*(choices.clone())).clone();
            }
            // if we're not at root, update choices
            if !current.parent_edge.is_none() {
                let parent = current.parent_edge.unwrap();
                choices.insert(parent.class, parent.node);
            }
            // if current is a leaf, then it's explored
            if current.to_visit.len() == 0 {
                current.explored = true;
            }
            // if all current's children are explored, then it's explored
            if current.edges.values().all(|n| n.explored) {
                current.explored = true;
            }
            // increment current's num_rollouts
            current.num_rollouts += 1; //TODO: is this right
            match current.parent {
                None => break,
                Some (n) => current = n
            }
        }
    }

    fn cost(&self, egraph: &EGraph, choices: Box<FxHashMap<ClassId, NodeId>>) -> f64 {
        let mut total_cost: f64 = 0.0;
        for (_, node_id) in choices.iter() {
            let node = egraph.nodes.get(node_id);
            if !node.is_none() {
                total_cost += node.unwrap().cost.into_inner();
            }
        }
        return total_cost;
    }
}

impl Extractor for MCTSExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        //TODO: Jacob
        let mut result = ExtractionResult::default();
        result.choices = IndexMap::new();
        let mcts_results = self.mcts(egraph, roots[0].clone(), NUM_ITERS);
        let size = roots.len();
        println!("roots size: {}",size);
        result.choices.extend(mcts_results.into_iter());
        println!("result size: {}",result.choices.len());
        // for root in roots {
        //     //TODO: multiple roots behavior?
        //     result.choices.extend(self.mcts(egraph, *root, NUM_ITERS).into_iter());
        // }
        return result;
    }
}
