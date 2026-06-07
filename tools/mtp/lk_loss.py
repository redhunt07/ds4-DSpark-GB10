"""LK Losses (arXiv 2602.23881, ICML 2026) — acceptance-direct loss for training
spec-decode draft heads. Replaces CE/KL.

Acceptance of a draft token under speculative verification is
    α = Σ_x min(p_x, q_x) = 1 − TV(p, q)
with p = target (base model) next-token dist, q = draft (MTP head) dist. KL is
only a proxy and diverges from α under capacity limits (our frozen-167M regime).
Two losses (the paper recommends the hybrid):
    L^α  = −log α                                  (acceptance NLL; ∇ = (1/α)∇TV)
    L^λ  = λ·KL(p‖q) + (1−λ)·TV(p,q),  λ = exp(−η·sg[α])
The per-step adaptive λ gives weak deep-step heads stronger KL guidance: low α ⇒
λ→1 (smooth KL signal), high α ⇒ λ→0 (direct TV/acceptance). α is aggregated to a
scalar per draft step (over the batch positions) for λ, per the paper.

Target p is SPARSE top-N (the base dist truncated + renormalized over its top-N,
as harvested — cf. MTP-D's Top-N=10000). q is the full draft softmax. Tokens
outside p's top-N have p=0, so they contribute 0 to α and KL; their q-mass folds
into TV's tail. This needs only log q at the N indices + q's mass on that support,
so it's O(N) not O(vocab). Hard one-hot p makes L^α = −log q(token) = CE — so LK
only differs from CE when p is a genuine soft target.
"""

import torch
import torch.nn.functional as F


def lk_terms(logits_q: torch.Tensor, p_idx: torch.Tensor, p_val: torch.Tensor):
    """logits_q [L, V] ; p_idx [L, N] long (vocab indices) ; p_val [L, N] (probs,
    ~sum 1 over top-N). -> (alpha [L], tv [L], kl [L]) per position.

    With p summing to 1 over top-N and q over the full vocab, α = Σ_topN min(p,q)
    equals 1 − TV exactly (TV = ½Σ_V|p−q|), since out-of-topN p=0 ⇒ |0−q|=q folds
    into the tail and the algebra collapses to Σ_topN min(p,q)."""
    logq = F.log_softmax(logits_q.float(), dim=-1)  # [L, V]
    logq_n = logq.gather(-1, p_idx)  # [L, N] log q at p's support
    q_n = logq_n.exp()  # [L, N] q at p's support
    alpha = torch.minimum(p_val, q_n).sum(-1)  # Σ_topN min(p,q) = acceptance
    tv = 1.0 - alpha
    logp = p_val.clamp_min(1e-12).log()
    kl = (p_val * (logp - logq_n)).sum(-1)  # forward KL(p‖q) over top-N
    return alpha, tv, kl


def lk_alpha_loss(logits_q, p_idx, p_val, eps: float = 1e-6):
    """L^α = −log α (mean over positions). Gradient = (1/α)·∇TV — TV direction,
    KL-like magnitude (auto-restored as α→0)."""
    alpha, _, _ = lk_terms(logits_q, p_idx, p_val)
    return (-alpha.clamp_min(eps).log()).mean()


def lk_hybrid_loss(logits_q, p_idx, p_val, eta: float = 3.0):
    """L^λ = λ·KL + (1−λ)·TV, λ = exp(−η·sg[mean α]) (scalar per draft step)."""
    alpha, tv, kl = lk_terms(logits_q, p_idx, p_val)
    lam = torch.exp(-eta * alpha.mean().detach())  # per-step scalar, stop-grad
    return (lam * kl + (1.0 - lam) * tv).mean()


def margin_loss(logits_q, target, m: float = 2.0):
    """Hinge on RANK-1 separation: penalize when the target (greedy token) logit
    isn't above the runner-up by margin m. Sharpens the argmax toward the target
    where CE/LK (distribution-match) leave it at rank 2-4. Zero grad once target is
    rank-1 by m, so it focuses gradient on the near-miss cases — the top-1 gap the
    top-k ceiling probe found. logits in fp32 for a stable -inf scatter."""
    z = logits_q.float()
    tgt_logit = z.gather(1, target.unsqueeze(1)).squeeze(1)
    runner_up = z.scatter(1, target.unsqueeze(1), float("-inf")).max(dim=1).values
    return torch.clamp(m - (tgt_logit - runner_up), min=0).mean()


LOSSES = {"lk_alpha": lk_alpha_loss, "lk_hybrid": lk_hybrid_loss}


def _selftest():
    torch.manual_seed(0)
    V, L, N = 512, 64, 16
    # build a soft target p over top-N indices, renormalized to sum 1
    p_idx = torch.stack([torch.randperm(V)[:N] for _ in range(L)])  # [L,N]
    p_raw = torch.rand(L, N).pow(3) + 1e-3
    p_val = p_raw / p_raw.sum(-1, keepdim=True)

    # (1) perfect match: q logits that reproduce p exactly -> α≈1, tv≈0
    q_perfect = torch.full((L, V), -30.0)
    q_perfect.scatter_(1, p_idx, p_val.clamp_min(1e-9).log())
    a, tv, kl = lk_terms(q_perfect, p_idx, p_val)
    assert a.mean() > 0.99 and tv.mean() < 0.01 and kl.mean() < 0.02, (
        f"perfect-match α={a.mean():.3f} tv={tv.mean():.3f} kl={kl.mean():.3f}"
    )

    # (2) hard one-hot p -> L^α reduces to CE = −log q(token)
    oh_idx = torch.randint(0, V, (L, 1))
    oh_val = torch.ones(L, 1)
    logits = torch.randn(L, V)
    la = lk_alpha_loss(logits, oh_idx, oh_val)
    ce = F.cross_entropy(logits, oh_idx.squeeze(1))
    assert torch.allclose(la, ce, atol=1e-4), f"L^α {la:.4f} != CE {ce:.4f}"

    # (3) adaptive λ: low α -> λ≈1 (KL); high α -> λ≈0 (TV)
    lo = torch.exp(-3.0 * torch.tensor(0.05))
    hi = torch.exp(-3.0 * torch.tensor(0.95))
    assert lo > 0.8 and hi < 0.1, f"λ(0.05)={lo:.2f} λ(0.95)={hi:.2f}"

    # (4) optimizing L^α drives α up toward 1 from a random draft
    q = torch.randn(L, V, requires_grad=True)
    opt = torch.optim.Adam([q], lr=0.2)
    a0 = lk_terms(q, p_idx, p_val)[0].mean().item()
    for _ in range(200):
        opt.zero_grad()
        loss = lk_alpha_loss(q, p_idx, p_val)
        loss.backward()
        opt.step()
    a1 = lk_terms(q, p_idx, p_val)[0].mean().item()
    assert a1 > 0.9 and a1 > a0 + 0.3, f"α {a0:.3f} -> {a1:.3f} (should rise to ~1)"

    print("lk_loss selftest OK:")
    print(f"  perfect-match α={a.mean():.4f} tv={tv.mean():.4f} kl={kl.mean():.4f}")
    print(f"  L^α==CE on one-hot: {float(la):.4f}")
    print(f"  λ(α=.05)={float(lo):.3f}  λ(α=.95)={float(hi):.3f}")
    print(f"  optimize L^α: α {a0:.3f} -> {a1:.3f}")


if __name__ == "__main__":
    _selftest()
