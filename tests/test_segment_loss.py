import torch

from autoevidence.training.segment_loss import segment_weighted_loss


def test_segment_loss_gradients():
    torch.manual_seed(0)
    B, L, V = 2, 8, 16
    logits = torch.randn(B, L, V, requires_grad=True)
    labels = torch.randint(0, V, (B, L))
    labels[:, :2] = -100
    w = torch.ones(B, L)
    w[:, :2] = 0
    w[:, 2:5] = 1.9
    w[:, 5:] = 1.3
    loss = segment_weighted_loss(logits, labels, w)
    loss.backward()
    assert logits.grad is not None
    assert float(loss.detach()) > 0
