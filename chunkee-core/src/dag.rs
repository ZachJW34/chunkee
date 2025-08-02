use petgraph::graph::{Graph, NodeIndex};
use std::collections::HashMap;

use crate::{coords::ChunkVector, grid::neighbors_of};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Task {
    LoadLOD0(ChunkVector),
    MeshLOD0(ChunkVector),
}

pub struct TaskGraph {
    graph: Graph<Task, ()>,
    task_nodes: HashMap<Task, NodeIndex>,
}

impl TaskGraph {
    pub fn new() -> Self {
        Self {
            graph: Default::default(),
            task_nodes: Default::default(),
        }
    }

    pub fn request_mesh(&mut self, cv: ChunkVector) {
        let mesh_node = self.add_task_if_new(Task::MeshLOD0(cv));
        let load_node = self.add_task_if_new(Task::LoadLOD0(cv));

        self.graph.add_edge(load_node, mesh_node, ());

        for neighbor_cv in neighbors_of(cv) {
            let load_neighbor_node = self.add_task_if_new(Task::LoadLOD0(neighbor_cv));
            self.graph.add_edge(load_neighbor_node, mesh_node, ());
        }
    }

    fn add_task_if_new(&mut self, task: Task) -> NodeIndex {
        if let Some(&node_index) = self.task_nodes.get(&task) {
            return node_index;
        }
        let node_index = self.graph.add_node(task);
        self.task_nodes.insert(task, node_index);
        node_index
    }

    fn remove_task(&mut self) {}
}
