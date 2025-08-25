import torch
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]
    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        for s, p in zip(self.shadow, self.params):
            p.data.copy_(s.data)
