use super::*;
use rand::seq::SliceRandom;
use std::collections::{BinaryHeap, HashSet};
use std::f64::consts::SQRT_2;
use std::mem::size_of;
use crate::faster_bottom_up::FasterBottomUpExtractor;
use crate::global_greedy_dag::GlobalGreedyDagExtractor;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
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
enum ExtractorType {
    Tree,
    Dag
}
type MCTSTree = Vec<MCTSNode>;
const EXPLORATION_PARAM: f64 = SQRT_2;
const NUM_ITERS: i64 = 50000;

fn result_dump(choices: &IndexMap<ClassId, NodeId>, egraph: &EGraph) -> () {
    println!("MCTS Choices: ");
    println!(
        "{}",
        choices
            .iter()
            .map(|(&ref key, &ref val)| format!("{:?}{:?}", key, val))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn pretty_print_tree(tree: &MCTSTree, index: usize, indent: usize) {
    if index >= tree.len() {
        return;
    }
    let node = &tree[index];

    // Print the current node with indentation
    for _ in 0..indent {
        print!("  ");
    }
    println!(
        "Node(min_cost: {:.2}, num_rollouts: {}, explored: {}, index: {})",
        node.min_cost, node.num_rollouts, node.explored, index
    );

    // Print each edge and its associated choice and child node
    for (choice, child_index) in &node.edges {
        for _ in 0..(indent + 1) {
            print!("  ");
        }
        println!("Choice {},{} ", choice.class, choice.node);

        // Recursive call to pretty print child nodes
        pretty_print_tree(tree, *child_index, indent + 1);
    }

    for _ in 0..indent {
        print!("  ");
    }
    println!("}}");
}
pub struct MCTSExtractor;
impl MCTSExtractor {
    fn mcts(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
        num_iters: i64,
        warm_start_extractor: Option<ExtractorType>
    ) -> FxHashMap<ClassId, NodeId> {
        // initialize the vector which will contain all our nodes
        let mut tree: MCTSTree = Vec::new();
        // warm start the tree
        println!("Size of MCTSNode: {}", size_of::<MCTSNode>());
        self.warm_start(warm_start_extractor, egraph, roots, &mut tree);
        let mut j = 0;
        for _ in 0..num_iters {
            j += 1;
            let leaf: Option<usize> = self.choose_leaf(0, egraph, &tree);
            match leaf {
                Some(leaf_index) => match self.rollout(leaf_index, egraph, &mut tree) {
                    None => continue,
                    Some((choices, new_node_index)) => {
                        self.backprop(egraph, new_node_index, choices, &mut tree)
                    }
                },
                None => {
                    break;
                }
            };
        }
        //pretty_print_tree(&tree, 0, 0);
        if j >= num_iters - 1 {
            println!("Timeout");
        }
        return tree[0].min_cost_map.clone();
    }

    fn warm_start(
        &self,
        warm_start_extractor: Option<ExtractorType>,
        egraph: &EGraph,
        roots: &[ClassId],
        tree_slot: &mut MCTSTree
    ) -> () {
        tree_slot.clear();
        // Initialize tree with default root node
        tree_slot.push(
            MCTSNode {
                to_visit: HashSet::from_iter(roots.iter().cloned()),
                decided_classes: FxHashMap::<ClassId, NodeId>::with_capacity_and_hasher(
                    egraph.classes().len(),
                    Default::default(),
                ),
                num_rollouts: 1,
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
                explored: false
            }
        );
        if warm_start_extractor.is_none() { return }
        // get extraction result from extractor
        let extraction_result: ExtractionResult = match warm_start_extractor.unwrap() {
            ExtractorType::Tree => FasterBottomUpExtractor {}.extract(egraph, roots),
            ExtractorType::Dag => GlobalGreedyDagExtractor {}.extract(egraph, roots)
        };
        // println!("{:?}", extraction_result.choices);
        // add results to tree
        let mut curr_index = 0;
        loop {
            // println!("curr_index: {curr_index}");
            if curr_index == tree_slot.len() { break }
            // For each e_class in to_visit:
            for eclass in tree_slot[curr_index].to_visit.clone().iter() {
                let chosen_node: &NodeId = &extraction_result.choices[eclass];
                let enode: &Node = &egraph[chosen_node];
                // decided_classes = curr.decided_classes[chosen_node/eclass]
                let mut decided_classes = tree_slot[curr_index].decided_classes.clone();
                decided_classes.insert((*eclass).clone(), (*chosen_node).clone());
                // to_vist is (curr.to_visit - eclass) U enode.children - decided
                let mut to_visit = tree_slot[curr_index].to_visit.clone();
                to_visit.remove(eclass);
                to_visit.extend(
                    enode.children.iter().map(|nid| (*egraph.nid_to_cid(nid)).clone())
                );
                for decided_class in decided_classes.keys() { to_visit.remove(decided_class); }
                // Add an edge to curr
                let new_node_index = tree_slot.len();
                tree_slot[curr_index].edges.insert(
                    MCTSChoice { class: (*eclass).clone(), node: (*chosen_node).clone()},
                    new_node_index
                );
                // Add a new MCTSNode
                // println!("pre push curr_index: {curr_index}, tree length: {}", new_node_index);
                tree_slot.push(
                    MCTSNode {
                        to_visit,
                        decided_classes,
                        num_rollouts: 1,
                        min_cost: 0.0,
                        min_cost_map: FxHashMap::<ClassId, NodeId>::with_capacity_and_hasher(
                            0,
                            Default::default()
                        ),
                        edges: FxHashMap::<MCTSChoice, usize>::with_capacity_and_hasher(
                            egraph.classes().len(),
                            Default::default(),
                        ),
                        parent: Some(curr_index),
                        parent_edge: Some(
                            MCTSChoice{ class: (*eclass).clone(), node: (*chosen_node).clone() }
                        ),
                        explored: false
                    }
                );
                println!("post push curr_index: {curr_index}, tree length: {}, tree capacity: {}", new_node_index, tree_slot.capacity());
            }
            curr_index += 1;
        }
        // extraction_result may contain extra, unused eclasses
        // the decided classes of the last added only holds the eclasses that we're using
        // copy it to be the min_cost_map of every node in our tree
        let choices = tree_slot.last().unwrap().decided_classes.clone();
        let cost = self.cost(egraph, Box::new(choices.clone()));
        for node in tree_slot {
            node.min_cost_map = choices.clone();
            node.min_cost = cost;
        }
    }

    fn choose_leaf(&self, curr: usize, egraph: &EGraph, tree: &MCTSTree) -> Option<usize> {
        // look for a choice not in curr's edges
        let curr_node = &tree[curr];
        for class_id in curr_node.to_visit.iter() {
            for node_id in egraph.classes().get(class_id).unwrap().nodes.iter()
            // .filter(|n| {
            //     !self.is_cycle(
            //         egraph,
            //         n,
            //         &curr_node.decided_classes.keys().collect::<HashSet<_>>(),
            //     )
            // })
            {
                let choice = MCTSChoice {
                    class: (*class_id).clone(),
                    node: (*node_id).clone(),
                };
                if !curr_node.edges.contains_key(&choice) {
                    return Some(curr);
                }
            }
        }
        // if we get here, then all choices are in curr's edges
        // filter edges for ones that are not explored
        let unexplored_children: Vec<usize> = curr_node
            .edges
            .values()
            .map(|i| (i, &tree[*i]))
            .filter(|(_, n)| !n.explored)
            .map(|(i, _)| *i)
            .collect();
        // if there are no unexplored children, then we've completely explored the tree
        if unexplored_children.is_empty() {
            return None;
        }
        // map nodes to uct cost and recurse on node which maximizes uct
        let next_curr =
            self.uct_choose_from_nodes(unexplored_children, curr_node.num_rollouts, tree);
        self.choose_leaf(next_curr, egraph, tree)
    }
    fn uct_choose_from_nodes(
        &self,
        node_indices: Vec<usize>,
        parent_num_rollouts: i32,
        tree: &MCTSTree,
    ) -> usize {
        // pre-compute min and max cost
        let mut min_cost: f64 = f64::INFINITY;
        let mut max_cost: f64 = f64::NEG_INFINITY;
        for node_index in node_indices.iter() {
            let node = &tree[*node_index];
            min_cost = min_cost.min(node.min_cost);
            max_cost = max_cost.max(node.min_cost)
        }
        // Initialize an empty max heap
        let mut uct_heap: BinaryHeap<(NotNan<f64>, usize)> =
            BinaryHeap::with_capacity(node_indices.len());
        // For each node, compute uct and insert into heap
        for (i, node_index) in node_indices.iter().enumerate() {
            let node = &tree[*node_index];
            let uct_cost = self.compute_uct(
                node.min_cost,
                min_cost,
                max_cost,
                node.num_rollouts as f64,
                parent_num_rollouts as f64,
            );
            uct_heap.push((NotNan::new(uct_cost).unwrap(), i))
        }
        let (_, i) = uct_heap.pop().unwrap();
        node_indices[i]
    }
    fn compute_uct(
        &self,
        cost: f64,
        min_cost: f64,
        max_cost: f64,
        num_rollouts: f64,
        parent_num_rollouts: f64,
    ) -> f64 {
        let cost_term = 1.0 - ((cost - min_cost) / (max_cost - min_cost));
        let rollout_term = EXPLORATION_PARAM * (parent_num_rollouts.ln() / num_rollouts).sqrt();
        if cost_term.is_nan() || rollout_term.is_nan() {
            return f64::NEG_INFINITY;
        }
        cost_term + rollout_term
    }
    fn find_first_node(
        &self,
        node_index: usize,
        egraph: &EGraph,
        tree: &MCTSTree,
    ) -> Option<MCTSChoice> {
        let node = &tree[node_index];
        for class_id in node.to_visit.iter() {
            for node_id in egraph.classes().get(class_id).unwrap().nodes.iter()
            // .filter(|n| {
            //     !self.is_cycle(
            //         egraph,
            //         n,
            //         &node.decided_classes.keys().collect::<HashSet<_>>(),
            //     )
            // })
            {
                if !node.edges.contains_key(&MCTSChoice {
                    class: (*class_id).clone(),
                    node: (*node_id).clone(),
                }) {
                    return Some(MCTSChoice {
                        class: (*class_id).clone(),
                        node: (*node_id).clone(),
                    });
                }
            }
        }
        return None;
    }
    fn is_cycle(
        &self,
        egraph: &EGraph,
        node_id: &NodeId,
        decided_classes: &HashSet<&ClassId>,
    ) -> bool {
        egraph[node_id].children.iter().any(|c| {
            *egraph.nid_to_cid(c) == egraph[node_id].eclass
                || decided_classes.contains(egraph.nid_to_cid(c))
        })
    }
    fn rollout(
        &self,
        node_index: usize,
        egraph: &EGraph,
        tree: &mut MCTSTree,
    ) -> Option<(Box<FxHashMap<ClassId, NodeId>>, usize)> {
        // get an MCTSChoice to take from the leaf node
        let first_choice: MCTSChoice = self
            .find_first_node(node_index.clone(), egraph, tree)
            .unwrap();

        // initialize the MCTSNode's decided map and add the MCTSChoice we chose above
        let mut new_decided: FxHashMap<ClassId, NodeId> = tree[node_index].decided_classes.clone();
        new_decided.insert(first_choice.class.clone(), first_choice.node.clone());

        // initialize a set of e-classes we need to visit
        let mut new_to_visit = tree[node_index].to_visit.clone();
        // remove the e-class corresponding to the choice made above
        new_to_visit.remove(&first_choice.class);
        // insert the e-classes of the children of the node we chose above
        let children = egraph[&first_choice.node]
            .children
            .clone()
            .into_iter()
            .map(|n| egraph.nid_to_cid(&n))
            .filter(|n| !new_decided.contains_key(n));
        for child in children {
            new_to_visit.insert(child.clone());
        }
        tree.push(MCTSNode {
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
            explored: false,
        });
        let new_node_index = tree.len() - 1;
        // update the leaf node to have an edge pointing to the node we just created
        tree[node_index]
            .edges
            .insert(first_choice.clone(), new_node_index);

        // clone new_to_visit to have a todo list for our rollout
        let mut todo = new_to_visit.clone();
        // initialize a map of choices taken on this rollout
        let mut choices: Box<FxHashMap<ClassId, NodeId>> = Box::new(new_decided.clone());

        while !todo.is_empty() {
            //randomly choose a class from todo, and a node from the class
            let to_visit_vec: Vec<ClassId> = todo.clone().into_iter().collect();
            let class_choice = to_visit_vec.choose(&mut rand::thread_rng()).unwrap();
            let eligible_nodes = egraph[class_choice]
                .nodes
                .iter()
                //.filter(|n| !self.is_cycle(egraph, n, &choices.keys().collect::<HashSet<_>>()))
                .collect::<Vec<_>>();
            let node_choice = match eligible_nodes.choose(&mut rand::thread_rng()) {
                Some(choice) => choice,
                None => {
                    return None;
                }
            };
            let choice = MCTSChoice {
                class: (*class_choice).clone(),
                node: (*node_choice).clone(),
            };
            //add choice to choices and add children of choice to todo
            choices.insert(choice.class.clone(), choice.node.clone());
            let children = egraph[&choice.node]
                .children
                .clone()
                .into_iter()
                .map(|n| egraph.nid_to_cid(&n))
                .filter(|n| !choices.contains_key(n));
            for child in children {
                todo.insert(child.clone());
            }
            todo.remove(&choice.class);
        }
        return Some((choices, new_node_index));
    }

    fn backprop(
        &self,
        egraph: &EGraph,
        mut current_index: usize,
        mut choices: Box<FxHashMap<ClassId, NodeId>>,
        tree: &mut MCTSTree,
    ) -> () {
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
            if tree[current_index]
                .edges
                .values()
                .all(|n| tree[*n].explored)
                && tree[current_index].edges.len() > 0
                && self.find_first_node(current_index, egraph, tree).is_none()
            {
                tree[current_index].explored = true;
            }
            // increment current's num_rollouts
            tree[current_index].num_rollouts += 1;
            match tree[current_index].parent {
                None => break,
                Some(n) => current_index = n,
            }
        }
    }

    // TODO: can choices just be a reference to a map?
    fn cost(&self, egraph: &EGraph, choices: Box<FxHashMap<ClassId, NodeId>>) -> f64 {
        let mut temp_result = ExtractionResult::default();
        temp_result.choices = IndexMap::new();
        temp_result.choices.extend(choices.clone().into_iter());
        if !temp_result
            .find_cycles(egraph, &egraph.root_eclasses)
            .is_empty()
        {
            return f64::INFINITY;
        }
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
        let mcts_results = self.mcts(egraph, roots, NUM_ITERS, Some(ExtractorType::Dag));
        let size = roots.len();
        result.choices.extend(mcts_results.into_iter());
        result_dump(&result.choices, egraph);
        if result.choices.is_empty() {
            let initial_result =
                super::faster_greedy_dag::FasterGreedyDagExtractor.extract(egraph, roots);
            log::info!("Unfinished MCTS solution");
            return initial_result;
        }
        return result;
    }
}
