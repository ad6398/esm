import torch

def backbone_distance_loss(
    X_pred: torch.Tensor,
    X_true: torch.Tensor
) -> torch.Tensor:
    """
    Compute the 'backbone_distance_loss' in batch:
      - X_pred, X_true: (B, L, 3, 3)
      - B = batch size
      - L = number of backbone transforms/residues
      - each item is a 3x3 matrix

    Steps (Algorithm 10):
      1) Flatten each 3x3 matrix to length 9 => (B, L, 9)
      2) For each batch element, compute pairwise squared distances 
         => D_pred, D_true of shape (B, L, L)
      3) Take the squared difference (D_pred - D_true)^2
      4) Clamp at 25
      5) Mean over all entries
    """
    B, L, _, _ = X_pred.shape
    
    # 1) Flatten: from (B, L, 3, 3) -> (B, L, 9)
    Xp = X_pred.view(B, L, 9)
    Xt = X_true.view(B, L, 9)

    # 2) Pairwise squared distances within each batch
    #    xp_diff has shape (B, L, L, 9) => sum along last dim => (B, L, L)
    xp_diff = Xp.unsqueeze(2) - Xp.unsqueeze(1)
    D_pred  = xp_diff.pow(2).sum(dim=-1)  # shape = (B, L, L)

    xt_diff = Xt.unsqueeze(2) - Xt.unsqueeze(1)
    D_true  = xt_diff.pow(2).sum(dim=-1)  # shape = (B, L, L)

    # 3) Elementwise squared difference
    E = (D_pred - D_true).pow(2)

    # 4) Clamp at 25
    E = torch.clamp(E, max=25.0)

    # 5) Mean over all batch entries and pairwise indices
    return E.mean()



def compute_vectors(X: torch.Tensor) -> torch.Tensor:
    """
    Given backbone coordinates X of shape (B, L, 3, 3),
    where each residue has [N, CA, C] in the 3rd dimension,
    returns the 6 direction vectors for each residue:

      (B, L, 6, 3)

    Vectors:
      v1 = (N -> CA)
      v2 = (CA -> C)
      v3 = (C -> N_next)
      v4 = - (v1 x v2)
      v5 = (C_prev->N) x (v1)
      v6 = (v2) x (v3)
    """
    # X shape: (B, L, 3, 3)
    # Each residue: N = X[...,0,:], CA = X[...,1,:], C = X[...,2,:]

    N  = X[:, :, 0, :]  # (B, L, 3)
    CA = X[:, :, 1, :]  # (B, L, 3)
    C  = X[:, :, 2, :]  # (B, L, 3)

    # v1 = CA - N
    v1 = CA - N
    # v2 = C - CA
    v2 = C - CA

    # We will zero-pad for next_N if there's no next residue
    next_N = torch.zeros_like(N)
    # for all residues except the last, the next_N is N of (i+1)
    next_N[:, :-1, :] = N[:, 1:, :]

    # v3 = (C -> N_next)
    v3 = next_N - C

    # v4 = - (v1 x v2)
    v4 = - torch.cross(v1, v2, dim=-1)

    # For the previous residue’s C (zero-padded at start):
    prev_C = torch.zeros_like(C)
    prev_C[:, 1:, :] = C[:, :-1, :]

    # v0 = (C_prev -> N)
    v0 = N - prev_C

    # v5 = (C_prev->N) x (v1)
    v5 = torch.cross(v0, v1, dim=-1)

    # v6 = (v2) x (v3) = (CA->C) x (C->N_next)
    v6 = torch.cross(v2, v3, dim=-1)

    # Stack to (B, L, 6, 3)
    return torch.stack([v1, v2, v3, v4, v5, v6], dim=2)


def backbone_direction_loss(X_pred: torch.Tensor, X_true: torch.Tensor) -> torch.Tensor:
    """
    X_pred, X_true: (B, L, 3, 3), each with (N,CA,C) coords for L residues.
    
    Implements the "backbone_direction_loss" by:
      1) computing 6 direction vectors for each residue
      2) computing all-pairs dot products
      3) taking squared difference, clamp at 20, mean
    """
    # 1. Compute the 6 vectors per residue
    V_pred = compute_vectors(X_pred)  # (B, L, 6, 3)
    V_true = compute_vectors(X_true)  # (B, L, 6, 3)

    # 2. Flatten from (B, L, 6, 3) to (B, 6L, 3)
    B, L = V_pred.shape[0], V_pred.shape[1]
    V_pred = V_pred.view(B, L*6, 3)
    V_true = V_true.view(B, L*6, 3)

    # 3. Pairwise dot-products in each batch
    # D_pred[i,j] = dot(V_pred[i], V_pred[j])
    # => shape (B, 6L, 6L)
    D_pred = torch.matmul(V_pred, V_pred.transpose(-1, -2))
    D_true = torch.matmul(V_true, V_true.transpose(-1, -2))

    # 4. Squared difference, clamp at 20
    E = (D_pred - D_true).pow(2)
    E = torch.clamp(E, max=20.0)

    # 5. Mean over all batch entries and pairwise indices
    return E.mean()

    