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
    edges: FxHashMap<MCTSChoice, usize>,
    parent: Option<usize>,
    parent_edge: Option<MCTSChoice>,
    explored: bool,
}
type MCTSTree = Vec<MCTSNode>;
const EXPLORATION_PARAM: f64 = SQRT_2;
const NUM_ITERS: i32 = 100000;

pub struct MCTSExtractor;
impl MCTSExtractor {
    //helper function
    // fn compute_mcts_tree_size(&self, root: &Box<MCTSNode>) -> i32{
    //     let mut sum: i32 = 1;
    //     for (_, node) in root.edges.iter(){
    //         sum += self.compute_mcts_tree_size(node);
    //     }
    //     return sum;
    // }

    fn mcts(&self, egraph: &EGraph, root: ClassId, num_iters: i32) -> FxHashMap<ClassId, NodeId> {
        // initialize the vector which will contain all our nodes
        let mut tree: MCTSTree = Vec::new();
        // initialize the root node of the tree
        tree.push(MCTSNode {
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
            edges: FxHashMap::<MCTSChoice, usize>::with_capacity_and_hasher(
                egraph.classes().len(),
                Default::default(),
            ),
            parent: None,
            parent_edge: None,
            explored: false,
        });
        for _ in 0..num_iters {
            let leaf: Option<usize> = self.choose_leaf(0, egraph, &tree);
            match leaf {
                Some(leaf_index) => {
                    match self.rollout(leaf_index, egraph, &mut tree){
                        None => continue,
                        Some ((choices, new_node_index)) =>
                         self.backprop(egraph, new_node_index, choices, &mut tree)
                    }
                    // println!("Tree size: {}",self.compute_mcts_tree_size(&root_node.clone()));
                    
                }
                None => break,
            };
        }
        return tree[0].min_cost_map.clone();
    }

    fn choose_leaf(&self, curr: usize, egraph: &EGraph, tree: &MCTSTree) -> Option<usize> {
        // look for a choice not in curr's edges
        let curr_node = &tree[curr];
        for class_id in curr_node.to_visit.iter() {
            //let good_nodes :Vec<_> = egraph.classes().get(class_id).unwrap().nodes.iter().filter(|n| self.is_self_cycle(egraph, n)).collect();
            for node_id in egraph.classes().get(class_id).unwrap().nodes.iter().filter(|n| !self.is_cycle(egraph, n,&curr_node.decided_classes.keys().collect::<HashSet<_>>())) {
                let choice = MCTSChoice {class: (*class_id).clone(), node: (*node_id).clone()};
                if !curr_node.edges.contains_key(&choice) {
                    return Some(curr);
                }
            }
        }
        // if we get here, then all choices are in curr's edges
        // filter edges for ones that are not explored
        let unexplored_children: Vec<usize> = curr_node.edges.values()
            .map(|i| (i, &tree[*i]))
            .filter(|(_, n)| !n.explored)
            .map(|(i, _)| *i)
            .collect();
        // if there are no unexplored children, then we've completely explored the tree
        if unexplored_children.is_empty() { return None }
        // map nodes to uct cost and recurse on node which maximizes uct
        let next_curr = self.uct_choose_from_nodes(unexplored_children, curr_node.num_rollouts, tree);
        self.choose_leaf(next_curr, egraph, tree)
    }
    fn uct_choose_from_nodes(&self, node_indices: Vec<usize>, parent_num_rollouts: i32, tree: &MCTSTree) -> usize {
        // pre-compute min and max cost
        let mut min_cost: f64 = f64::INFINITY;
        let mut max_cost: f64 = f64::NEG_INFINITY;
        for node_index in node_indices.iter() {
            let node = &tree[*node_index];
            min_cost = min_cost.min(node.min_cost);
            max_cost = max_cost.max(node.min_cost)
        }
        // Initialize an empty max heap
        let mut uct_heap: BinaryHeap<(NotNan<f64>, usize)> = BinaryHeap::with_capacity(node_indices.len());
        // For each node, compute uct and insert into heap
        for (i, node_index) in node_indices.iter().enumerate() {
            let node = &tree[*node_index];
            let uct_cost = self.compute_uct(node.min_cost, min_cost, max_cost, node.num_rollouts as f64, parent_num_rollouts as f64);
            uct_heap.push((NotNan::new(uct_cost).unwrap(), i))
        }
        let (_, i) = uct_heap.pop().unwrap();
        node_indices[i]
    }
    fn compute_uct(&self, cost: f64, min_cost: f64, max_cost: f64, num_rollouts: f64, parent_num_rollouts: f64) -> f64 {
        let cost_term = 1.0 - ((cost - min_cost) / (max_cost - min_cost));
        let rollout_term = EXPLORATION_PARAM * (parent_num_rollouts.ln() / num_rollouts).sqrt();
        cost_term + rollout_term
    }
    fn find_first_node(& self, node_index: usize, egraph: & EGraph,tree: &MCTSTree) -> Option<MCTSChoice> {
        let node = &tree[node_index];
        for class_id in node.to_visit.iter(){
            for node_id in egraph.classes().get(class_id).unwrap().nodes.iter()
                .filter(|n|!self.is_cycle(egraph, n,&node.decided_classes.keys().collect::<HashSet<_>>())){
                if !node.edges.contains_key(&MCTSChoice { class: (*class_id).clone(), node: (*node_id).clone() }){
                    return Some(MCTSChoice { class: (*class_id).clone(), node: (*node_id).clone() });
                }
            }
        }
        return None;
    }
    fn is_cycle(&self, egraph: &EGraph, node_id: &NodeId, decided_classes: &HashSet<&ClassId>) -> bool{
        egraph[node_id].children.iter().any(|c| *egraph.nid_to_cid(c) == egraph[node_id].eclass ||
             decided_classes.contains(egraph.nid_to_cid(c)))
    }
    fn rollout(&self, node_index: usize, egraph: &EGraph, tree: &mut MCTSTree) -> Option<(Box<FxHashMap<ClassId, NodeId>>, usize)> {
        // get an MCTSChoice to take from the leaf node
        let first_choice : MCTSChoice = self.find_first_node(node_index.clone(),egraph,tree).unwrap();

        // initialize the MCTSNode's decided map and add the MCTSChoice we chose above
        let mut new_decided : FxHashMap<ClassId, NodeId> = tree[node_index].decided_classes.clone();
        new_decided.insert(first_choice.class.clone(),first_choice.node.clone());
        
        // initialize a set of e-classes we need to visit
        let mut new_to_visit = tree[node_index].to_visit.clone();
        // remove the e-class corresponding to the choice made above
        new_to_visit.remove(&first_choice.class);
        // insert the e-classes of the children of the node we chose above
        let children = egraph[&first_choice.node].children.clone().into_iter()
            .map(|n| egraph.nid_to_cid(&n))
            .filter(|n| !new_decided.contains_key(n));
        for child in children{
            new_to_visit.insert(child.clone());
        }
        tree.push(MCTSNode{
            to_visit: new_to_visit.clone(),
            decided_classes: new_decided.clone(),
            num_rollouts: 0,
            min_cost: f64::INFINITY,
            min_cost_map: FxHashMap::<ClassId, NodeId>::with_capacity_and_hasher(
                egraph.classes().len(),
                Default::default(),
            ),
            edges: FxHashMap::<MCTSChoice, usize>::with_capacity_and_hasher(
                egraph.classes().len(),
                Default::default(),
            ),
            parent: Some(node_index),
            parent_edge: Some(first_choice.clone()),
            explored: false
        });
        let new_node_index = tree.len() - 1;
        // update the leaf node to have an edge pointing to the node we just created
        tree[node_index].edges.insert(first_choice.clone(), new_node_index);

        // clone new_to_visit to have a todo list for our rollout
        let mut todo = new_to_visit.clone();
        // initialize a map of choices taken on this rollout
        let mut choices : Box<FxHashMap<ClassId, NodeId>> = Box::new(new_decided.clone());

        while !todo.is_empty(){
            //randomly choose a class from todo, and a node from the class
            let to_visit_vec : Vec<ClassId> = todo.clone().into_iter().collect();
            let class_choice = to_visit_vec.choose(&mut rand::thread_rng()).unwrap();
            let eligible_nodes = egraph[class_choice].nodes
                .iter()
                .filter(|n| !self.is_cycle(egraph,n,&choices.keys().collect::<HashSet<_>>()))
                .collect::<Vec<_>>();
            let node_choice = match eligible_nodes.choose(&mut rand::thread_rng()) {
                    Some (choice) => choice,
                    None => {
                        tree.remove(new_node_index);
                        tree[node_index].edges.remove(&first_choice.clone());
                        return None;
                    }
                };
            let choice = MCTSChoice { class: (*class_choice).clone(), node: (*node_choice).clone() };
            //add choice to choices and add children of choice to todo
            choices.insert(choice.class.clone(),choice.node.clone());
            let children = egraph[&choice.node]
                .children
                .clone()
                .into_iter()
                .map(|n| egraph.nid_to_cid(&n))
                .filter(|n| !choices.contains_key(n));
            for child in children{
                todo.insert(child.clone());
            }
            todo.remove(&choice.class);
        }
        return Some((choices, new_node_index));
    }

    fn backprop(&self, egraph: &EGraph, mut current_index: usize, mut choices: Box<FxHashMap<ClassId, NodeId>>, tree: &mut MCTSTree) -> () {
        loop {
            // compute cost
            let choices_cost = self.cost(egraph, choices.clone());
            // update choices & cost of current node
            if choices_cost < tree[current_index].min_cost {
                tree[current_index].min_cost = choices_cost;
                tree[current_index].min_cost_map = (*(choices.clone())).clone();
            }
            // if we're not at root, update choices
            if !tree[current_index].parent_edge.is_none() {
                let parent = tree[current_index].parent_edge.clone().unwrap();
                choices.insert(parent.class, parent.node);
            }
            // if current is a leaf, then it's explored
            if tree[current_index].to_visit.len() == 0 {
                tree[current_index].explored = true;
            }
            // if all current's children are explored, then it's explored
            if tree[current_index].edges.values().all(|n| tree[*n].explored) {
                tree[current_index].explored = true;
            }
            // increment current's num_rollouts
            tree[current_index].num_rollouts += 1;
            match tree[current_index].parent {
                None => break,
                Some (n) => current_index = n
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
        let mut result = ExtractionResult::default();
        result.choices = IndexMap::new();
        let mcts_results = self.mcts(egraph, roots[0].clone(), NUM_ITERS);
        let size = roots.len();
        let mut vec: Vec<_> = egraph.nodes.iter().map(|(&ref key, &ref value)| (key, value.children.clone(), value.eclass.clone())).collect();
        vec.sort();
        //println!("MCTS, Nodes: {:?}",vec);
        result.choices.extend(mcts_results.into_iter());
        let vec: Vec<_> = result.choices.iter().map(|(&ref key, &ref value)| (key, value)).collect();
        //println!("choices: {:?}",vec);
        //println!("Roots: {:?}",roots);
        //println!("result size: {}",result.choices.len());
        // for root in roots {
        //     //TODO: multiple roots behavior?
        //     result.choices.extend(self.mcts(egraph, *root, NUM_ITERS).into_iter());
        // }
        return result;
    }
}
