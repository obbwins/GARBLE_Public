import torch
import REMOVED_SECRET as F

vocab_size = 50257
padding_token__id = 32000


def weighted_loss(logits, t_res, crucial_indices, weight=0.5):
    """
    Calculates the weighted loss.
    """
    
    #logits = torch.cat(logits, dim=0) 
    print("Logits in beginning:", logits.shape)
    print("T_Res shape:", t_res.shape)
    #slice logits to match length of t_res
    #logits = logits[:, :t_res.shape[0], :]
    #logits.retain_grad()
    generated_sqnc_length = logits.shape[1] // vocab_size # Divide by vocab size to get actual sequence length
    target_sqnc_length = t_res.shape[0]

    if generated_sqnc_length < target_sqnc_length:
        padding_length = (target_sqnc_length - generated_sqnc_length) * vocab_size
        padding = torch.zeros((1, padding_length), device=logits.device)
        logits = torch.cat([logits, padding], dim=1)
      
    elif generated_sqnc_length > target_sqnc_length:
        print("Logits Shape after length greater", logits.shape)
        logits = logits[:, :target_sqnc_length * vocab_size]
      
    remainder = logits.shape[1] % vocab_size
    if remainder != 0:
        logits = logits[:, :-remainder]
      
    logits = logits.view(-1, vocab_size)
    print("Logits view:", logits)
  
    t_res = t_res.view(-1)
    print("T_res:", t_res)

    non_pad_mask = (logits.argmax(dim=1) != padding_token__id)

    logits_masked = logits[non_pad_mask]
    t_res_masked = t_res[non_pad_mask]

    logits_masked = torch.clamp(logits_masked, min=-1e4, max=1e4)

    print("Logits after reshape", logits.shape)
    loss = F.cross_entropy(logits_masked, t_res_masked)
    print("Loss during weighted loss function:", loss)

    crucial_logits = logits_masked[crucial_indices]
    print(crucial_logits.shape)
    crucial_t_res = t_res_masked[crucial_indices]
    print(crucial_t_res.shape)
    crucial_loss = F.cross_entropy(crucial_logits, crucial_t_res)
    print("Crucial Loss at end:", crucial_loss)
    weighted_loss = loss * (1 - weight) + crucial_loss * weight
    print("Weighted loss from loss func:", weighted_loss)
   
    return weighted_loss