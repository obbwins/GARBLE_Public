import torch
import REMOVED_SECRET as F

vocab_size = 50257
padding_token_id = 32000

def safe_log(x):
    return torch.log(torch.clamp(x, min=1e-40))

def label_smoothed_nll_loss(lprobs, target, epsilon=0.1, ignore_index=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    nll_loss = nll_loss.squeeze(-1)
    smooth_loss = smooth_loss.squeeze(-1)
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss.mean()


def weighted_loss(logits, t_res, crucial_indices, weight=0.5, epsilon = 0.1):
    """
    Calculates the weighted loss.
    """
    print("Logits shape:", logits.shape)
    print("T_Res shape:", t_res.shape)
    print("Crucial indices:", crucial_indices)

    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("Warning: NaN or Inf values in logits")
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

    # Reshape logits if necessary
    if logits.dim() == 2:
        pass
    elif logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))

    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    
    # Ensure t_res is the correct shape
    t_res = t_res.view(-1).long()

    # Pad or truncate logits to match t_res length
    target_length = t_res.shape[0]
    current_length = logits.shape[0]
    
    if current_length < target_length:
        logits = F.pad(logits, (0, 0, 0, target_length - current_length))
    elif current_length > target_length:
        logits = logits[:target_length]

    print("Sample logits:", logits[:5, :5])
    print("Sample t_res:", t_res[:5])


    # Apply label smoothing
    smoothed_targets = torch.full_like(logits, epsilon / (vocab_size - 1))
    smoothed_targets.scatter_(1, t_res.unsqueeze(1), 1 - epsilon)

    #Apply label smoothing
    lprobs = F.log_softmax(logits, dim=-1)
    loss = -(smoothed_targets * lprobs).sum(dim=-1).mean()   

    # Compute the crucial loss
    if crucial_indices:
        crucial_logits = logits[crucial_indices]
        crucial_t_res = t_res[crucial_indices]
        crucial_smoothed_targets = smoothed_targets[crucial_indices]
        crucial_lprobs = F.log_softmax(crucial_logits, dim=-1)
        crucial_loss = -(crucial_smoothed_targets * crucial_lprobs).sum(dim=-1).mean()
    else:
        crucial_loss = loss  # If no crucial indices, use the main loss
        
    # Compute the weighted loss
    weighted_loss = loss * (1 - weight) + crucial_loss * weight

    print("Main Loss:", loss.item())
    print("Crucial Loss:", crucial_loss.item())
    print("Weighted Loss:", weighted_loss.item())

    return weighted_loss