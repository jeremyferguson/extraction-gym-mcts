use std::cmp::max;
use super::*;
use crate::faster_bottom_up::FasterBottomUpExtractor;
use crate::faster_greedy_dag::FasterGreedyDagExtractor;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{BTreeSet, BinaryHeap, HashSet};
use std::f64::consts::SQRT_2;
use std::ops::{Index, IndexMut};

#[derive(PartialEq, Eq, Hash, Clone, Debug, Ord, PartialOrd)]
struct MCTSChoice {
    class: ClassId,
    node: NodeId,
}
#[derive(Clone)]
struct MCTSNode {
    to_visit: FxHashSet<ClassId>,
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
    Dag,
}
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct MCTSTreeKey {
    decided_classes: BTreeSet<MCTSChoice>,
    to_visit: BTreeSet<ClassId>,
}
struct MCTSTree {
    keys: FxHashMap<MCTSTreeKey, usize>,
    nodes: Vec<MCTSNode>,
}
impl MCTSTree {
    fn len(&self) -> usize {
        self.nodes.len()
    }
    fn push(&mut self, node: MCTSNode, key: MCTSTreeKey) -> usize {
        // returns the index of the inserted node
        self.keys.insert(key, self.len());
        self.nodes.push(node);
        self.len() - 1
    }
}
impl Index<usize> for MCTSTree {
    type Output = MCTSNode;
    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}
impl IndexMut<usize> for MCTSTree {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}
const EXPLORATION_PARAM: f64 = SQRT_2;
const NUM_ITERS: i64 = 10000;
const WARMSTART_LIMIT: usize = 1000;
const P_RAND_ROLLOUT: f64 = 0.2;
const MEMORY_LIMIT: usize = 14 * 1024 * 1024 * 1024; // 14 GB

fn result_dump(choices: &IndexMap<ClassId, NodeId>, _egraph: &EGraph) -> () {
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
fn check_memory_usage() -> Result<(), String> {
    match memory_stats::memory_stats() {
        Some(mem_stats) => {
            if max(mem_stats.physical_mem, mem_stats.virtual_mem) > MEMORY_LIMIT {
                Err("Memory usage approaching limit".to_string())
            } else {
                Ok(())
            }
        }
        None => {
            Err("Couldn't get memory usage stats".to_string())
        }
    }
}

pub struct MCTSExtractor;
impl MCTSExtractor {
    fn mcts(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
        num_iters: i64,
        warm_start_extractor: Option<ExtractorType>,
    ) -> FxHashMap<ClassId, NodeId> {
        // initialize the vector which will contain all our nodes
        let mut tree: MCTSTree = MCTSTree {
            keys: HashMap::with_capacity_and_hasher(roots.len(), Default::default()),
            nodes: Vec::new(),
        };
        // warm start the tree
        self.warm_start(warm_start_extractor, egraph, roots, &mut tree);
        println!("Finished warm start");
        let dag_solution = super::faster_greedy_dag::FasterGreedyDagExtractor
            .extract(egraph, roots)
            .choices;
        let mut j = 0;
        //pretty_print_tree(&tree, 0, 0);
        for _ in 0..num_iters {
            match check_memory_usage(){
                Ok (())=> (),
                Err (_) => {println!("Reached memory cap");return tree.nodes[0].min_cost_map.clone()}
            }
            j += 1;
            let leaf: Option<usize> = self.choose_leaf(0, egraph, &mut tree);
            match leaf {
                Some(leaf_index) => {
                    match self.rollout(leaf_index, egraph, &mut tree, &dag_solution) {
                        None => continue,
                        Some((choices, new_node_index)) => {
                            self.backprop(egraph, new_node_index, choices, &mut tree)
                        }
                    }
                }
                None => {
                    break;
                }
            };
        }
        //pretty_print_tree(&tree, 0, 0);
        if j >= num_iters - 1 {
            println!("Timeout");
        }
        return tree.nodes[0].min_cost_map.clone();
    }

    fn warm_start(
        &self,
        warm_start_extractor: Option<ExtractorType>,
        egraph: &EGraph,
        roots: &[ClassId],
        tree_slot: &mut MCTSTree,
    ) -> () {
        tree_slot.keys.clear();
        tree_slot.nodes.clear();
        // Initialize tree with default root node
        let root_key = MCTSTreeKey {
            decided_classes: BTreeSet::new(),
            to_visit: roots.iter().map(|cid| (*cid).clone()).collect(),
        };
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
                explored: false,
            },
            root_key,
        );
        if warm_start_extractor.is_none() {
            return;
        }
        // get extraction result from extractor
        let extraction_result: ExtractionResult = match warm_start_extractor.unwrap() {
            ExtractorType::Tree => FasterBottomUpExtractor {}.extract(egraph, roots),
            ExtractorType::Dag => FasterGreedyDagExtractor {}.extract(egraph, roots),
        };
        // println!("{:?}", extraction_result.choices);
        // add results to tree
        let mut curr_index;
        // add nodes via dfs
        let mut dfs_visited: FxHashSet<usize> =
            HashSet::with_capacity_and_hasher(roots.len(), Default::default());
        let mut dfs_to_visit: Vec<usize> = Vec::new();
        dfs_to_visit.push(0);
        // tracked used choices of extraction result to compute cost later
        let mut used_choices: FxHashMap<ClassId, NodeId> =
            HashMap::with_capacity_and_hasher(roots.len(), Default::default());
        'outer: loop {
            // println!("curr_index: {curr_index}");
            if dfs_to_visit.is_empty() {
                break;
            }
            if dfs_visited.len() > WARMSTART_LIMIT { break }
            curr_index = dfs_to_visit.pop().unwrap();
            if dfs_visited.contains(&curr_index) {
                continue;
            }
            // For each e_class in to_visit:
            for eclass in tree_slot.nodes[curr_index].to_visit.clone().iter() {
                // if tree_slot.len() > WARMSTART_LIMIT {
                //     break 'outer;
                // }
                let chosen_node = &extraction_result.choices[eclass];
                let enode = &egraph[chosen_node];
                used_choices.insert(eclass.clone(), chosen_node.clone());
                // decided_classes = curr.decided_classes[chosen_node/eclass]
                let mut decided_classes = tree_slot.nodes[curr_index].decided_classes.clone();
                decided_classes.insert((*eclass).clone(), (*chosen_node).clone());
                // to_vist is (curr.to_visit - eclass) U enode.children - decided
                let mut to_visit = tree_slot.nodes[curr_index].to_visit.clone();
                to_visit.remove(eclass);
                to_visit.extend(
                    enode
                        .children
                        .iter()
                        .map(|nid| (*egraph.nid_to_cid(nid)).clone()),
                );
                for decided_class in decided_classes.keys() {
                    to_visit.remove(decided_class);
                }
                // if to_visit.is_empty() {
                //     println!("Empty to visit found!")
                // }
                // check whether node that would be created already exists
                let new_node_key = MCTSTreeKey {
                    decided_classes: decided_classes
                        .iter()
                        .map(|(cid, nid)| MCTSChoice {
                            class: cid.clone(),
                            node: nid.clone(),
                        })
                        .collect(),
                    to_visit: to_visit.iter().map(|cid| cid.clone()).collect(),
                };
                let new_node_index = match tree_slot.keys.get(&new_node_key) {
                    Some(i) => *i,
                    // create a new MCTSNode if needed
                    None => tree_slot.push(
                        MCTSNode {
                            to_visit,
                            decided_classes,
                            num_rollouts: 1,
                            min_cost: 0.0,
                            min_cost_map: FxHashMap::<ClassId, NodeId>::with_capacity_and_hasher(
                                0,
                                Default::default(),
                            ),
                            edges: FxHashMap::<MCTSChoice, usize>::with_capacity_and_hasher(
                                egraph.classes().len(),
                                Default::default(),
                            ),
                            parent: Some(curr_index),
                            parent_edge: Some(MCTSChoice {
                                class: (*eclass).clone(),
                                node: (*chosen_node).clone(),
                            }),
                            explored: false,
                        },
                        new_node_key,
                    ),
                };
                // Add an edge to curr
                tree_slot.nodes[curr_index].edges.insert(
                    MCTSChoice {
                        class: (*eclass).clone(),
                        node: (*chosen_node).clone(),
                    },
                    new_node_index,
                );
                dfs_to_visit.push(new_node_index);
                // println!(
                //     "post push curr_index: {curr_index}, tree length: {}, tree capacity: {}",
                //     tree_slot.len(),
                //     tree_slot.nodes.capacity()
                // );
            }
            dfs_visited.insert(curr_index);
        }
        // extraction_result may contain extra, unused eclasses
        // copy used_choices to be the min_cost_map of every node in our tree
        println!("Used choices length: {:?}", used_choices.len());
        println!(
            "Extraction result choices: {:?}",
            extraction_result.choices.len()
        );
        let mut cost = self.cost(egraph, Box::new(used_choices.clone()));
        if cost.is_finite() {
            for node in &mut tree_slot.nodes {
                node.min_cost_map = used_choices.clone();
                node.min_cost = cost;
            }
        } else {
            cost = self.cost(
                egraph,
                Box::new(FxHashMap::from_iter(extraction_result.choices.clone().into_iter()))
            );
            for node in &mut tree_slot.nodes {
                node.min_cost_map = FxHashMap::from_iter(
                    extraction_result.choices.clone().into_iter()
                );
                node.min_cost = cost;
            }
        }
    }

    fn choose_leaf(&self, curr: usize, egraph: &EGraph, tree: &mut MCTSTree) -> Option<usize> {
        // look for a choice not in curr's edges
        let curr_to_visit = tree[curr].to_visit.clone();
        for class_id in curr_to_visit.iter() {
            for node_id in egraph.classes().get(class_id).unwrap().nodes.iter() {
                let choice = MCTSChoice {
                    class: (*class_id).clone(),
                    node: (*node_id).clone(),
                };
                if !tree[curr].edges.contains_key(&choice) {
                    // Check whether an equivalent node already exists in the Tree
                    let mut decided_classes: BTreeSet<MCTSChoice> = tree[curr]
                        .decided_classes
                        .iter()
                        .map(|(c, n)| MCTSChoice {
                            class: c.clone(),
                            node: n.clone(),
                        })
                        .collect();
                    decided_classes.insert(choice.clone());
                    let mut to_visit: BTreeSet<ClassId> =
                        tree[curr].to_visit.iter().cloned().collect();
                    to_visit.extend(
                        egraph[node_id]
                            .children
                            .iter()
                            .map(|n| egraph.nid_to_cid(n))
                            .cloned(),
                    );
                    to_visit =
                        &to_visit - &decided_classes.iter().map(|c| c.class.clone()).collect();
                    let key = MCTSTreeKey {
                        decided_classes,
                        to_visit,
                    };
                    let node_index_option = tree.keys.get(&key);
                    let node_index: usize;
                    match node_index_option {
                        // if one doesn't, return this node
                        None => {
                            return Some(curr);
                        }
                        // if one does, update curr_node's edges
                        Some(i) => {
                            node_index = *i;
                        }
                    }
                    tree[curr].edges.insert(choice, node_index);
                }
            }
        }
        // if we get here, then all choices are in curr's edges
        // filter edges for ones that are not explored
        let unexplored_children: Vec<usize> = tree[curr]
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
            self.uct_choose_from_nodes(unexplored_children, tree[curr].num_rollouts, tree);
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
            let node = &tree.nodes[*node_index];
            min_cost = min_cost.min(node.min_cost);
            max_cost = max_cost.max(node.min_cost)
        }
        // Initialize an empty max heap
        let mut uct_heap: BinaryHeap<(NotNan<f64>, usize)> =
            BinaryHeap::with_capacity(node_indices.len());
        // For each node, compute uct and insert into heap
        for (i, node_index) in node_indices.iter().enumerate() {
            let node = &tree.nodes[*node_index];
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
        tree: &mut MCTSTree,
    ) -> Option<MCTSChoice> {
        let curr_node = &mut tree.nodes[node_index];
        for class_id in curr_node.to_visit.iter() {
            for node_id in egraph.classes().get(class_id).unwrap().nodes.iter() {
                let choice = MCTSChoice {
                    class: (*class_id).clone(),
                    node: (*node_id).clone(),
                };
                if !curr_node.edges.contains_key(&choice) {
                    // Check whether an equivalent node already exists in the Tree
                    let mut decided_classes: BTreeSet<MCTSChoice> = curr_node
                        .decided_classes
                        .iter()
                        .map(|(c, n)| MCTSChoice {
                            class: c.clone(),
                            node: n.clone(),
                        })
                        .collect();
                    decided_classes.insert(choice.clone());
                    let mut to_visit: BTreeSet<ClassId> =
                        curr_node.to_visit.iter().cloned().collect();
                    to_visit.extend(
                        egraph[node_id]
                            .children
                            .iter()
                            .map(|n| egraph.nid_to_cid(n))
                            .cloned(),
                    );
                    to_visit =
                        &to_visit - &decided_classes.iter().map(|c| c.class.clone()).collect();
                    let key = MCTSTreeKey {
                        decided_classes,
                        to_visit,
                    };
                    match tree.keys.get(&key) {
                        // if it doesn't, return this choice
                        None => {
                            return Some(choice);
                        }
                        // if it does, update curr_node's edges
                        Some(i) => {
                            curr_node.edges.insert(choice, *i);
                        }
                    }
                }
            }
        }
        return None;
    }
    // fn is_cycle(
    //     &self,
    //     egraph: &EGraph,
    //     node_id: &NodeId,
    //     decided_classes: &HashSet<&ClassId>,
    // ) -> bool {
    //     egraph[node_id].children.iter().any(|c| {
    //         *egraph.nid_to_cid(c) == egraph[node_id].eclass
    //             || decided_classes.contains(egraph.nid_to_cid(c))
    //     })
    // }
    fn rollout(
        &self,
        node_index: usize,
        egraph: &EGraph,
        tree: &mut MCTSTree,
        oracle_solution: &IndexMap<ClassId, NodeId>,
    ) -> Option<(Box<FxHashMap<ClassId, NodeId>>, usize)> {
        // get an MCTSChoice to take from the leaf node
        let first_choice: MCTSChoice = self
            .find_first_node(node_index.clone(), egraph, tree)
            .unwrap();

        // initialize the MCTSNode's decided map and add the MCTSChoice we chose above
        let mut new_decided: FxHashMap<ClassId, NodeId> = tree[node_index].decided_classes.clone();
        new_decided.insert(first_choice.class.clone(), first_choice.node.clone());

        // initialize a set of e-classes we still need to visit
        let mut new_to_visit = tree[node_index].to_visit.clone();
        // remove the e-class corresponding to the choice made above
        new_to_visit.remove(&first_choice.class);
        // insert the e-classes of the children of the node we chose above
        let children = egraph[&first_choice.node]
            .children
            .iter()
            .map(|n| (*(egraph.nid_to_cid(&n))).clone())
            .filter(|n| !new_decided.contains_key(n));
        new_to_visit.extend(children);
        let new_node_index = tree.push(
            MCTSNode {
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
            },
            MCTSTreeKey {
                decided_classes: new_decided
                    .iter()
                    .map(|(c, n)| MCTSChoice {
                        class: c.clone(),
                        node: n.clone(),
                    })
                    .collect(),
                to_visit: new_to_visit.iter().cloned().collect(),
            },
        );
        // update the leaf node to have an edge pointing to the node we just created
        tree[node_index]
            .edges
            .insert(first_choice.clone(), new_node_index);
        // clone new_to_visit to have a todo list for our rollout
        let mut todo = new_to_visit.clone();
        // initialize a map of choices taken on this rollout
        let mut choices: Box<FxHashMap<ClassId, NodeId>> = Box::new(new_decided.clone());

        let mut rng = rand::thread_rng();
        while !todo.is_empty() {
            //randomly choose a class from todo, and a node from the class
            let mut to_visit_vec: Vec<ClassId>;
            let rand_rollout = rng.gen_bool(P_RAND_ROLLOUT);
            if rand_rollout {
                to_visit_vec = todo.clone().into_iter().collect();
            } else {
                to_visit_vec = todo
                    .clone()
                    .into_iter()
                    .filter(|c| oracle_solution.contains_key(c))
                    .collect();
                if to_visit_vec.is_empty() {
                    to_visit_vec = todo.clone().into_iter().collect();
                }
            }
            let class_choice = to_visit_vec.choose(&mut rng).unwrap();
            let eligible_nodes = egraph[class_choice].nodes.iter().collect::<Vec<_>>();
            let node_choice;
            let rand_choice = eligible_nodes.choose(&mut rand::thread_rng()).unwrap();
            let dag_choice = oracle_solution.get(class_choice);
            node_choice = match dag_choice {
                Some(choice) => {
                    if rand_rollout {
                        *rand_choice
                    } else {
                        &choice
                    }
                }
                None => *rand_choice,
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
            if self.find_first_node(current_index, egraph, tree).is_none()
                && tree[current_index].edges.len() > 0
                && tree[current_index]
                    .edges
                    .values()
                    .all(|n| tree[*n].explored)
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
        // let mut temp_result = ExtractionResult::default();
        // temp_result.choices = IndexMap::new();
        // temp_result.choices.extend(choices.clone().into_iter());
        // if !temp_result
        //     .find_cycles(egraph, &egraph.root_eclasses)
        //     .is_empty()
        // {
        //     return f64::INFINITY;
        // }
        match self.find_cycles(&choices, egraph, &egraph.root_eclasses) {
            Some(cycles) if cycles.is_empty() => (),
            None | Some(_) => return f64::INFINITY,
        }
        return *self.dag_cost(egraph, &egraph.root_eclasses, &choices);
        // let mut total_cost: f64 = 0.0;
        // for (_, node_id) in choices.iter() {
        //     let node = egraph.nodes.get(node_id);
        //     if !node.is_none() {
        //         total_cost += node.unwrap().cost.into_inner();
        //     }
        // }
        // return total_cost;
    }

    pub fn dag_cost(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
        choices: &FxHashMap<ClassId, NodeId>,
    ) -> Cost {
        let mut costs: IndexMap<ClassId, Cost> = IndexMap::new();
        let mut todo: Vec<ClassId> = roots.to_vec();
        while let Some(cid) = todo.pop() {
            let node_id = &choices[&cid];
            let node = &egraph[node_id];
            if costs.insert(cid.clone(), node.cost).is_some() {
                continue;
            }
            for child in &node.children {
                todo.push(egraph.nid_to_cid(child).clone());
            }
        }
        costs.values().sum()
    }

    pub fn find_cycles(
        &self,
        choices: &FxHashMap<ClassId, NodeId>,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> Option<Vec<ClassId>> {
        // let mut status = vec![Status::Todo; egraph.classes().len()];
        let mut status = IndexMap::<ClassId, Status>::default();
        let mut cycles = vec![];
        for root in roots {
            // let root_index = egraph.classes().get_index_of(root).unwrap();
            match self.cycle_dfs(egraph, root, &mut status, &mut cycles, choices) {
                Ok(()) => continue,
                Err(_) => return None,
            }
        }
        Some(cycles)
    }

    fn cycle_dfs(
        &self,
        egraph: &EGraph,
        class_id: &ClassId,
        status: &mut IndexMap<ClassId, Status>,
        cycles: &mut Vec<ClassId>,
        choices: &FxHashMap<ClassId, NodeId>,
    ) -> Result<(), String> {
        match status.get(class_id).cloned() {
            Some(Status::Done) => Ok(()),
            Some(Status::Doing) => {
                cycles.push(class_id.clone());
                return Ok(());
            }
            None => {
                status.insert(class_id.clone(), Status::Doing);
                if !choices.contains_key(class_id) {
                    return Err(format!("Class not found: {class_id}"));
                }
                let node_id = &choices[class_id];
                let node = &egraph[node_id];
                for child in &node.children {
                    let child_cid = egraph.nid_to_cid(child);
                    match self.cycle_dfs(egraph, child_cid, status, cycles, choices) {
                        Ok(_) => continue,
                        Err(e) => return Err(e),
                    }
                }
                status.insert(class_id.clone(), Status::Done);
                return Ok(());
            }
        }
    }
}

impl Extractor for MCTSExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        let mut result = ExtractionResult::default();
        result.choices = IndexMap::new();
        let warm_start_extractor = Some(ExtractorType::Dag);
        let mcts_results = self.mcts(egraph, roots, NUM_ITERS, warm_start_extractor);
        let size = roots.len();
        result.choices.extend(mcts_results.clone().into_iter());
        match self.find_cycles(&mcts_results, egraph, roots) {
            Some(cycles) if cycles.is_empty() => return result,
            None | Some(_) => {
                let initial_result =
                    super::faster_greedy_dag::FasterGreedyDagExtractor.extract(egraph, roots);
                println!("Unfinished MCTS solution");
                return initial_result;
            }
        }
    }
}
