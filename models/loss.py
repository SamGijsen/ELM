import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class ELM_e_FrozenLM_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, device, batch_size, gamma=0.5, DDP=False):
        # M-FLAG Loss from C. Liu ea. 2023, arXiv:2307.08347v2
        super(ELM_e_FrozenLM_Loss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.DDP = DDP
        self.gamma = gamma

    def orthogonal_loss(self, x1, x2):
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        logits = torch.mm(x1.T, x2).to(self.device)
        logits.div_(self.batch_size)
        on_diag = torch.diagonal(logits).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(logits).pow_(2).sum()
        loss = on_diag + 0.0051*off_diag
        return loss/2

    def align_loss(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        loss = 2 - 2 * (x * y).sum(dim=-1)
        loss += 2 - 2 * (y * x).sum(dim=-1)
        return loss.mean()
    
    def forward(self, eeg_emb, proj_eeg_emb, proj_text_emb):
        orthogonal_loss = self.orthogonal_loss(eeg_emb, eeg_emb) * self.gamma
        align_loss = self.align_loss(proj_eeg_emb, proj_text_emb) * (1-self.gamma)
        return orthogonal_loss, align_loss 
    
class ELM_el_FrozenLM_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, device, batch_size, DDP=False, temp=0.07):
        super(ELM_el_FrozenLM_Loss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.DDP = DDP
        self.temp = temp 
        
    def forward(self, proj_eeg_emb, proj_text_emb):
        if self.DDP:
            proj_eeg_emb = torch.cat(FullGatherLayer.apply(proj_eeg_emb), dim=0)
            proj_text_emb = torch.cat(FullGatherLayer.apply(proj_text_emb), dim=0)

        proj_eeg_emb = F.normalize(proj_eeg_emb, dim=1)
        proj_text_emb = F.normalize(proj_text_emb, dim=1)

        logits = (proj_text_emb @ proj_eeg_emb.T) / self.temp
        eeg_sim = proj_eeg_emb @ proj_eeg_emb.T
        text_sim = proj_text_emb @ proj_text_emb.T
        targets = F.softmax((eeg_sim + text_sim) / 2 * self.temp, dim=-1)
        
        text_loss = cross_entropy(logits, targets, reduction='none')
        eeg_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (eeg_loss + text_loss) / 2.0
        return loss.mean()
    
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

    
class ELM_MIL_FrozenLM_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, device, batch_size, DDP=False, temp=0.07, style='x,y',
                 max_eeg_pairs=32, max_text_pairs=8):
        super(ELM_MIL_FrozenLM_Loss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.DDP = DDP
        self.temp = temp
        self.alignment_style = style
        self.max_eeg_pairs = int(max_eeg_pairs)
        self.max_text_pairs = int(max_text_pairs)

    def compute_loss_x_given_y(self, sim_matrix, id1, id2):
        positive_mask = (id1.unsqueeze(1) == id2.unsqueeze(0))
        log_softmax_col = F.log_softmax(sim_matrix, dim=0)

        return -(positive_mask * log_softmax_col).sum(dim=0) / positive_mask.sum(dim=0)
    
    def compute_loss_y_given_x(self, sim_matrix, id1, id2):
        positive_mask = (id1.unsqueeze(1) == id2.unsqueeze(0))
        log_softmax_row = F.log_softmax(sim_matrix, dim=1)

        return -(positive_mask * log_softmax_row).sum(dim=1) / positive_mask.sum(dim=1)
    

    def forward(self, proj_eeg_emb, proj_text_emb, eeg_id, text_id, eeg_ix, eeg_sub_ix, text_ix):

        proj_eeg_emb = F.normalize(proj_eeg_emb, dim=-1)
        proj_text_emb = F.normalize(proj_text_emb, dim=-1)
        sim_matrix = (proj_eeg_emb @ proj_text_emb.T) / self.temp

        if self.alignment_style == 'y|x':
            loss = self.compute_loss_y_given_x(sim_matrix, eeg_id, text_id).mean()
        elif self.alignment_style == 'x|y':
            loss = self.compute_loss_x_given_y(sim_matrix, eeg_id, text_id).mean()
        elif self.alignment_style == 'x,y':
            ygx = self.compute_loss_y_given_x(sim_matrix, eeg_id, text_id).mean()
            xgy = self.compute_loss_x_given_y(sim_matrix, eeg_id, text_id).mean()
            loss = (ygx + xgy) / 2

        else:
            raise ValueError("Invalid alignment style")
        
        return loss

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
