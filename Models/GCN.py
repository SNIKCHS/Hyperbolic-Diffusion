import torch
import torch.nn as nn
from layers.layers import GCLayer
from schnetpack.nn import AtomDistances, HardCutoff
from schnetpack import Properties


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight,gain=0.25)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
class GCN(nn.Module):
    def __init__(self, device,args):
        super().__init__()
        self.device = device
        self.act = nn.SiLU()
        self.Layer = nn.Sequential(
            GCLayer( 20, 128,  self.act),
            GCLayer(128, 128, self.act),
            GCLayer(128, 128, self.act),
            GCLayer(128, 128, self.act),
            GCLayer(128, 128, self.act),
        )
        self.embedding = nn.Embedding(10, 20)
        self.distances = AtomDistances()
        self._edges_dict = {}
        self.out = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            # ResBlock(64),
            nn.Linear(64, 1)
        )
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.cutoff = HardCutoff()
        self.apply(weight_init)

    def forward(self, inputs):
        atomic_numbers = inputs[Properties.Z]  # (b,n_atom)
        positions = inputs[Properties.R]  # (b,n_atom,3)
        positions -= positions.mean(dim=1, keepdim=True)
        node_mask = inputs[Properties.atom_mask]  # (b,n_atom)
        u0 = inputs['energy_U0']
        batch_size, n_nodes = atomic_numbers.size()

        size = node_mask.size()
        edge_mask = node_mask.unsqueeze(2).expand(size[0], size[1], size[1])  # (b,n_atom,n_atom)
        edge_mask = edge_mask * edge_mask.permute(0, 2, 1)

        ar = torch.arange(atomic_numbers.size(1), device=atomic_numbers.device)[None, None, :].repeat(
            atomic_numbers.size(0), atomic_numbers.size(1), 1)  # (b,n_atom,n_atom)
        nbh = ar * edge_mask
        h = self.embedding(atomic_numbers)  # (b,n_atom,embed)

        distance = self.distances(positions, nbh.long(), neighbor_mask=edge_mask.bool())
        edges = self.get_adj_matrix(n_nodes, batch_size)

        h = h.view(batch_size * n_nodes, -1)

        distance = distance.view(batch_size * n_nodes * n_nodes, 1)
        node_mask = node_mask.view(batch_size * n_nodes, -1)
        edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
        edge_mask = self.cutoff(distance) * edge_mask

        input = (h, distance, edges, node_mask, edge_mask)
        output, distances, edges, node_mask, edge_mask = self.Layer(input)
        output = self.out(output) * node_mask
        output = output.view(batch_size, n_nodes).sum(1, keepdim=True)
        # print(output)
        # print(u0)
        loss = torch.sqrt(self.loss_fn(output, u0))
        MAE_loss = torch.nn.functional.l1_loss(output, u0,reduction='mean')
        return loss,MAE_loss

    def get_adj_matrix(self, n_nodes, batch_size):
        # 对每个n_nodes，batch_size只要算一次
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(self.device),
                         torch.LongTensor(cols).to(self.device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size)