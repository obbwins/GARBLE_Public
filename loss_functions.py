import torch
import REMOVED_SECRET as F

vocab_size = 50257
padding_token_id = 32000

def weighted_loss(logits, t_res, crucial_indices, weight=0.5):
    """
    Calculates the weighted loss.
    """
    print("Logits shape:", logits.shape)
    print("T_Res shape:", t_res.shape)

    # Reshape logits if necessary
    if logits.dim() == 3:
        logits = logits.view(-1, vocab_size)
    
    # Ensure t_res is the correct shape
    t_res = t_res.view(-1)

    # Pad or truncate logits to match t_res length
    target_length = t_res.shape[0]
    current_length = logits.shape[0]
    
    if current_length < target_length:
        logits = F.pad(logits, (0, 0, 0, target_length - current_length))
    elif current_length > target_length:
        logits = logits[:target_length]

    # Create a soft mask for non-padding tokens
    soft_mask = 1 - F.softmax(logits, dim=-1)[:, padding_token_id].unsqueeze(1)
    
    # Apply the soft mask
    logits_masked = logits * soft_mask
    
    # Compute the main loss
    loss = F.cross_entropy(logits_masked, t_res, reduction='none')
    loss = loss.mean()

    # Compute the crucial loss
    crucial_logits = logits_masked[crucial_indices]
    crucial_t_res = t_res[crucial_indices]
    crucial_loss = F.cross_entropy(crucial_logits, crucial_t_res, reduction='mean')

    # Compute the weighted loss
    weighted_loss = loss * (1 - weight) + crucial_loss * weight

    print("Main Loss:", loss.item())
    print("Crucial Loss:", crucial_loss.item())
    print("Weighted Loss:", weighted_loss.item())

    return weighted_loss