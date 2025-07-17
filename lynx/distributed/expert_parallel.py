import torch
from torch import Tensor, nn
from torch.nn import Linear
import torch.nn.functional as F
from torch import einsum
import numpy as np
import torch_xla.distributed.spmd.xla_sharding as xs
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import math
from torch_xla.distributed.spmd import Mesh
from transformers.activations import ACT2FN
def make_dispatched_expert_inputs_parallel(tensor: Tensor,mesh):
    """
    :param tensor: shape=[E, G, C, M]
    :return: None
    """
    xs.mark_sharding(tensor, mesh, ('expert', None, None, None)) 

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(Expert, self).__init__()
        self.fc1 = nn.Parameter(torch.empty(num_experts, hidden_dim,input_dim))
        self.fc2 = nn.Parameter(torch.empty(num_experts, input_dim,hidden_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=2)

class MoELayer(nn.Module):
    def __init__(self, config):
        super(MoELayer, self).__init__()
        self.experts = Expert(config.hidden_size, 
        config.intermediate_size, 
        config.num_experts)
        self.num_experts = config.num_experts
        self.gate = GatingNetwork(config.hidden_size, config.num_experts)
        self.act_fn = ACT2FN[config.hidden_act]
        self.mesh = config.mesh
    def Top2Gating(self, gating_scores):
        G, S, E = gating_scores.shape
        C = 2
        device = gating_scores.device
        # Get top-2 experts for each token
        top_k_gating, top_k_indices = torch.topk(gating_scores, k=C, dim=2)  # [G, S, C]
        
        # Normalize top-k weights
        top_k_gating = F.softmax(top_k_gating, dim=-1)
        # Initialize weights and masks
        combine_weights = torch.zeros((G, S, E, C), dtype=gating_scores.dtype,device=device)
        dispatch_mask   = torch.zeros((G, S, E, C), dtype=gating_scores.dtype,device=device)
        # Build indices for scatter operation
        # Need to expand [G, S, C] indices to [G, S, C, 1], then expand to [G, S, C, C]
        indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, C)  # [G, S, C, C]
        
        # Assign weights to top-k experts for each token
        combine_weights = combine_weights.scatter(
            dim=2,  # Fill in the expert dimension
            index=indices_expanded,  # [G, S, C, C]
            src=top_k_gating.unsqueeze(-1).expand(-1, -1, -1, C)  # [G, S, C, C]
        )
        
        # Assign dispatch mask to top-k experts for each token
        dispatch_mask = dispatch_mask.scatter(
            dim=2,
            index=indices_expanded,
            src=torch.ones((G, S, C, C), dtype=gating_scores.dtype,device=device)
        )
        # # Generate expert indices for round-robin assignment and assign directly
        # for g in range(G):
        #     for s in range(S):
        #         pos = g * S + s
        #         e1 = (pos * 2) % E
        #         e2 = (pos * 2 + 1) % E
        #         # Assign 0.5 to all "output channels" of these two experts
        #         combine_weights[g, s, e1, :] = 0.5
        #         combine_weights[g, s, e2, :] = 0.5
        #         # 同理 mask 赋 1
        #         dispatch_mask[g, s, e1, :] = 1.0
        #         dispatch_mask[g, s, e2, :] = 1.0

        return combine_weights, dispatch_mask

    
    def forward(self, x):
        # x: [G, S, M]
        gating_scores = self.gate(x)  # [G,S,E]
        combine_weights , dispatch_mask = self.Top2Gating(gating_scores)  #combine_weights,dispatch_mask形状都为[G,S,E,C]

        num_devices = xr.global_runtime_device_count()
        mesh_shape = (num_devices, 1)
        device_ids = np.arange(num_devices)
        # mesh = Mesh(device_ids, mesh_shape, ('data', 'model'))
        xs.mark_sharding(combine_weights, self.mesh, ('expert', None, None, None))
        xs.mark_sharding(dispatch_mask,   self.mesh, ('expert', None, None, None))


        dispatched_expert_inputs = einsum(
            'gsec,gsm->egcm', 
            dispatch_mask, 
            x
        )  # [E, G, C, M]
        
        make_dispatched_expert_inputs_parallel(dispatched_expert_inputs, self.mesh)

        wi = self.get_expert_parameters("fc1")  # [E, H, M]
        wo = self.get_expert_parameters("fc2")  # [E, M, H]
        h = einsum('egcm,ehm->egch', dispatched_expert_inputs, wi)  # [E, G, C, H]
        h = self.act_fn(h)

        expert_outputs = einsum('egch,emh->gecm', h, wo)  # [G, E, C, M]
        outputs = einsum('gsec,gecm->gsm', combine_weights, expert_outputs)  # [G, S, M]
        return outputs

    def get_expert_parameters(self, layer_name="fc1"):
        """
        get all experts' specified layer(fc1) weights, concatenate to a 3D matrix
        output shape: (num_experts, out_features, in_features)
        """
        return getattr(self.experts, layer_name)

def make_input_parallel(x:Tensor,mesh:xs.Mesh):
    xs.mark_sharding(x, mesh, ("data", None,None)) #data sharding

def make_expert_parallel(moelayer:MoELayer,mesh:xs.Mesh):

    fc1_weight = moelayer.get_expert_parameters("fc1")
    xs.mark_sharding(fc1_weight,mesh,('expert',None,None))
    fc2_weight = moelayer.get_expert_parameters("fc2")
    xs.mark_sharding(fc2_weight,mesh,('expert',None,None))


def make_gating_network_parallel(gating_network:GatingNetwork,mesh:xs.Mesh):
    xs.mark_sharding(gating_network.gate.weight, mesh, ('data',None))
class LlamaConfig(object):
    hidden_size = 4096
    intermediate_size = 11008
    hidden_act = "gelu"
    num_experts = 16
if __name__ == "__main__":
    xr.use_spmd()
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    expert_parallel = 2
    
    expert_parallel_shape = (
        expert_parallel,
        num_devices//expert_parallel,
    )
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, expert_parallel_shape, 
                ('expert', 'data'))

    # dimension parameters
    config = LlamaConfig()
    G = 2  # number of groups
    S = 3  # sequence length
    E = config.num_experts  # number of experts
    M = config.hidden_size  # input dimension
    H = config.intermediate_size # hidden dimension
    # C = 2  # Top2 capacity

    # input data
    inputs = torch.randn(G, S, M).to(device)  # [G, S, M]
    make_input_parallel(inputs, mesh)
    moelayer = MoELayer(config, mesh)
    moelayer.to(device)
    make_expert_parallel(moelayer,mesh)
    make_gating_network_parallel(moelayer.gate,mesh)
    
    for i in range(5):
        moelayer.zero_grad()
        xm.mark_step()
        outputs = moelayer(inputs)
        loss = outputs.sum()
        loss.backward()
        print("outputs shape:", outputs.shape)
        xm.mark_step()
