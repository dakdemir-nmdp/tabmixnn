import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Family:
    def __init__(self):
        pass

    @property
    def n_params(self):
        raise NotImplementedError

    def forward_params(self, eta):
        """Transform raw neural output eta to distribution parameters theta."""
        raise NotImplementedError

    def nll(self, theta, y, **kwargs):
        """Compute Negative Log Likelihood."""
        raise NotImplementedError

    def predict(self, theta, mode="default", **kwargs):
        raise NotImplementedError
    
    def validate_targets(self, df, target_cols):
        pass

class Gaussian(Family):
    def __init__(self, link="identity"):
        super().__init__()
        self.link = link

    @property
    def n_params(self):
        # mean and log_std (or we can treat std as global param, but here let's predict it for full flexibility or use 1 if fixed)
        # Blueprint says: mu=eta, optionally sigma=exp(delta) (global or predicted). 
        # Let's assume simpler case: 1 output for mu, sigma is a learnable scaler handled elsewhere or we output 2 params.
        # For this implementation, let's output 2 params: mu and log_sigma for maximum flexibility
        return 2 

    def forward_params(self, eta):
        # eta is [batch, 2] -> mu, log_sigma
        mu = eta[:, 0]
        log_sigma = eta[:, 1]
        return mu, log_sigma

    def nll(self, theta, y, **kwargs):
        mu = theta[:, 0]
        log_sigma = theta[:, 1]
        # y might be shape [batch] or [batch, 1]
        y = y.squeeze()
        sigma = torch.exp(log_sigma)
        return -torch.distributions.Normal(mu, sigma).log_prob(y).mean()

    def predict(self, theta, mode="default", **kwargs):
        mu = theta[:, 0]
        return mu.cpu().numpy()

class Binomial(Family):
    def __init__(self, link="logit"):
        super().__init__()
        self.link = link

    @property
    def n_params(self):
        return 1

    def forward_params(self, eta):
        return eta.squeeze(-1) # Logits

    def nll(self, theta, y, **kwargs):
        # theta is logits
        # y is 0/1
        return F.binary_cross_entropy_with_logits(theta.squeeze(-1), y.float().squeeze(), reduction='mean')

    def predict(self, theta, mode="default", **kwargs):
        probs = torch.sigmoid(theta)
        if mode == "proba":
            return probs.cpu().numpy()
        return (probs > 0.5).long().cpu().numpy()

    def predict_proba(self, theta, **kwargs):
        return torch.sigmoid(theta).cpu().numpy()

class Poisson(Family):
    def __init__(self):
        super().__init__()

    @property
    def n_params(self):
        return 1

    def forward_params(self, eta):
        # eta is log_rate
        return eta.squeeze(-1)

    def nll(self, theta, y, **kwargs):
        # theta is log_rate
        return F.poisson_nll_loss(theta, y.float().squeeze(), log_input=True, reduction='mean')

    def predict(self, theta, mode="default", **kwargs):
        return torch.exp(theta).cpu().numpy()

class NegBinomial(Family):
    def __init__(self):
        super().__init__()

    @property
    def n_params(self):
        return 2 # log_mu, log_alpha

    def forward_params(self, eta):
        return eta[:, 0], eta[:, 1]

    def nll(self, theta, y, **kwargs):
        # Based on NB2 parameterization
        # mu = exp(eta_mu), alpha = exp(eta_alpha)
        # Var = mu + alpha * mu^2
        # PyTorch doesn't have a direct NegBin NLL with this parameterization usually, 
        # but we can implement it via lgamma.
        # Or use torch.distributions.NegativeBinomial
        # PyTorch NegBin uses total_count (r) and probs (p) or logits.
        # Mean = r(1-p)/p. 
        # We need mapping. 
        # mu = r(1-p)/p
        # alpha = 1/r => r = 1/alpha
        log_mu = theta[:, 0]
        log_alpha = theta[:, 1]
        mu = torch.exp(log_mu)
        alpha = torch.exp(log_alpha)
        
        # r = 1/alpha
        r = 1.0 / (alpha + 1e-6)
        
        # p = r / (r + mu)
        p = r / (r + mu)
        
        # Create distribution
        try:
            dist = torch.distributions.NegativeBinomial(total_count=r, probs=p)
            nll = -dist.log_prob(y.squeeze()).mean()
            return nll
        except ValueError:
            # Fallback or numerical stability
            return torch.tensor(0.0, requires_grad=True, device=mu.device)

    def predict(self, theta, mode="default", **kwargs):
        log_mu = theta[:, 0]
        return torch.exp(log_mu).cpu().numpy()

class DiscreteTimeSurvival(Family):
    def __init__(self, time_bins):
        super().__init__()
        self.time_bins = np.array(time_bins)
        
    @property
    def n_params(self):
        return len(self.time_bins)

    def forward_params(self, eta):
        # eta: [n, T] logits for hazard
        return eta

    def nll(self, theta, y, **kwargs):
        # theta: logits h_t
        # y: expected to have 'time_idx' and 'event' (d)
        # But here y might be just passed as tensor. 
        # We need extra info. 
        # The blueprint says nll(theta, y, **extra).
        # We assume y is tuple (time_idx, event_status) or tensor with both.
        # Let's assume y is [batch, 2] where col 0 is time_idx (0..T-1) and col 1 is event (0/1).
        
        time_idx = y[:, 0].long()
        event = y[:, 1].float()
        
        # Construct mask
        # For each subject i, we care about t <= time_idx[i]
        # h_{it} = sigmoid(theta_{it})
        
        # We need to compute log likelihood efficiently.
        # L_i = sum_{t < T_i} log(1 - h_{it}) + d_i * log(h_{iT_i}) + (1-d_i) * log(1-h_{iT_i})?
        # Blueprint: - sum_{t < t_i} log(1-h_{it}) - d_i log(h_{it_i}) - (1-d_i) log(1-h_{it_i}) (if censored, the last term is last survival prob)
        # Actually blueprint eq 9.1:
        # if event: sum_{t < t_i} log(1-h_{it}) + log(h_{it_i})
        # if censored at t_i: sum_{t <= t_i} log(1-h_{it})
        
        # Let's vectorize.
        batch_size, T = theta.shape
        
        # Range [0, 1, ..., T-1]
        t_range = torch.arange(T, device=theta.device).unsqueeze(0) # [1, T]
        
        # Mask for terms before observed time: t < time_idx
        mask_before = (t_range < time_idx.unsqueeze(1)).float()
        
        # Mask for term at observed time: t == time_idx
        mask_at = (t_range == time_idx.unsqueeze(1)).float()
        
        h = torch.sigmoid(theta)
        # numerical stability check
        h = torch.clamp(h, 1e-7, 1 - 1e-7)
        
        # Log terms
        log_1_minus_h = torch.log(1 - h)
        log_h = torch.log(h)
        
        # Contribution from survival up to time_idx (obs or censored)
        # For uncensored (d=1) & censored (d=0) we always survive all t < time_idx
        # loss_part1 = - sum (mask_before * log(1-h))
        loss_surv_before = -torch.sum(mask_before * log_1_minus_h, dim=1)
        
        # Contribution at time_idx
        # If event (d=1): need h_{i t_i} => - log(h)
        # If censored (d=0): need 1 - h_{i t_i} => - log(1-h)  (i.e., survived this interval too)
        
        # d_i * (-log h) + (1-d_i) * (-log(1-h))
        loss_at = event * (-torch.sum(mask_at * log_h, dim=1)) + \
                  (1 - event) * (-torch.sum(mask_at * log_1_minus_h, dim=1))
                  
        return (loss_surv_before + loss_at).mean()

    def predict(self, theta, mode="survival", **kwargs):
        # theta are logits
        h = torch.sigmoid(theta)
        surv_cond = 1 - h
        surv = torch.cumprod(surv_cond, dim=1)
        # Prepend 1.0 at t=0? Blueprint says S(t) = prod_{s<=t} (1-h_s)
        # This gives S at end of bin.
        if mode == "survival":
            return surv.detach().cpu().numpy()
        return h.detach().cpu().numpy()

class DiscreteTimeCompetingRisks(Family):
    def __init__(self, time_bins, num_causes):
        super().__init__()
        self.time_bins = np.array(time_bins)
        self.num_causes = num_causes # K

    @property
    def n_params(self):
        # logits for each cause at each time
        return len(self.time_bins) * self.num_causes

    def forward_params(self, eta):
        # reshape eta to [batch, T, K]
        # Blueprint 9.2: eta_{itk}
        n_bins = len(self.time_bins)
        return eta.reshape(-1, n_bins, self.num_causes)

    def nll(self, theta, y, **kwargs):
        # theta: [batch, T, K] logits
        # y: [batch, 2], time_idx and cause (0=censored, 1..K=causes)
        
        time_idx = y[:, 0].long()
        cause = y[:, 1].long() # 0 = censored, >0 event type
        
        batch_size, T, K = theta.shape
        
        # p_{itk} = exp(eta) / (1 + sum exp(eta))
        # p_{it0} = 1 / (1 + sum exp(eta))
        
        # We can treat this as K+1 classes multinomial at each time step where class 0 is "no event yet"
        # However, the "no event" prob matches 1 / (1+sum...)
        # We can concat a zero logit for the "no event" class if we want to use log_softmax
        # Logits: [batch, T, K]
        # Pad with 0 for the "no-event" class (implicit 0 logit)
        
        zeros = torch.zeros(batch_size, T, 1, device=theta.device)
        all_logits = torch.cat([zeros, theta], dim=2) # [batch, T, K+1] where idx 0 is 'no event'
        
        log_probs = F.log_softmax(all_logits, dim=2) # log p_{itc}, c in 0..K
        
        # t_range [1, T]
        t_range = torch.arange(T, device=theta.device).unsqueeze(0)
        
        mask_before = (t_range < time_idx.unsqueeze(1)).float()
        mask_at = (t_range == time_idx.unsqueeze(1)).float()
        
        # If t < t_i, we observed outcome 0 (did not fail by any cause)
        # loss += - log p_{it0}
        log_p0 = log_probs[:, :, 0]
        loss_before = -torch.sum(mask_before * log_p0, dim=1)
        
        # At t == t_i:
        # If cause = k > 0: observed k. loss += - log p_{itk}
        # If cause = 0 (censored): observed 0. loss += - log p_{it0}
        
        # We can gather the log prob of the observed cause at each time step
        # But we only care about the specific cause at time_idx
        
        # Broadcast cause to [batch, T] just to select easily? No.
        
        # Let's get log probs at time_idx
        # dimension 1: time_idx
        # efficient gathering
        # But `mask_at` is easier.
        
        # We want log prob of class `cause` at `time_idx`.
        # Construct target class indices for 'at' step:
        # If censored/event, the class index is `cause` (0..K)
        
        # log_probs: [batch, T, K+1]
        # We want to sum -log_prob[batch, t, 0] for t < time_idx
        # And -log_prob[batch, time_idx, cause] 
        
        # One liner validation:
        # If censored (cause=0), we want p_{it0} at t=time_idx too? 
        # Blueprint says: if censored at t_tilde: -sum_{t <= t_tilde} log(p_{it0})
        # My logic above covers this because cause=0 implies selecting index 0 at time_idx.
        
        # For the 'at' step:
        # We need log_probs[., ., cause]
        # One-hot encode cause? 
        # cause is [batch]
        
        # manual indexing
        # Select the log_probs for the specific causes at the specific times is tricky with pure masking if we want vectorized
        # But we can calculate loss for all classes and mask.
        
        # log_probs_at_cause = Gather log_probs at index `cause`
        # index [batch, T] -> cause repeated? No cause is only valid at time_idx.
        # But for t < time_idx, the observed "class" was 0.
        
        # Let's build a target matrix Y_target of shape [batch, T] with class indices.
        # Initialize with 0.
        # Set Y_target[i, time_idx[i]] = cause[i]
        # But wait, for t > time_idx, what is it? We don't care (masked out).
        
        Y_target = torch.zeros(batch_size, T, dtype=torch.long, device=theta.device)
        # Use scatter to set the cause at time_idx
        # time_idx is [batch]
        # We need to clamp time_idx to avoid out of bounds if time_idx >= T? 
        # Assuming time_idx is valid 0..T-1.
        
        # We only set value if time_idx < T. If time_idx is T (fully survived?), we might treat differently.
        # Assuming max time_idx is T-1.
        
        Y_target.scatter_(1, time_idx.unsqueeze(1), cause.unsqueeze(1))
        
        # Now gather log probs
        # log_probs [batch, T, K+1]
        # gather on dim 2
        picked_log_probs = log_probs.gather(2, Y_target.unsqueeze(2)).squeeze(2) # [batch, T]
        
        # Now apply masks
        # For t < time_idx: we want Y_target=0 (which it is), so picked_log_probs is log p_{it0}
        # For t == time_idx: we want Y_target=cause, so picked_log_probs is log p_{it,cause}
        
        mask_relevant = mask_before + mask_at
        
        loss = -torch.sum(mask_relevant * picked_log_probs, dim=1)
        return loss.mean()

    def predict(self, theta, mode="cif", **kwargs):
        # theta: [batch, T, K]
        # CIF calculation
        batch_size, T, K = theta.shape
        zeros = torch.zeros(batch_size, T, 1, device=theta.device)
        all_logits = torch.cat([zeros, theta], dim=2)
        probs = F.softmax(all_logits, dim=2) # [batch, T, K+1]
        
        # p_{itk} is prob of event k at time t, given survival to t-1
        # S(t-1) probability of surviving up to t-1
        # Incidence I(t, k) = S(t-1) * p_{itk}
        # CIF(t, k) = sum_{s<=t} I(s, k)
        
        # p_{it0} is prob of surviving interval t given survival to t-1
        p_surv_interval = probs[:, :, 0] # [batch, T]
        
        # S(t) = S(t-1) * p_{it0}
        # Cumulative product
        # S(-1) = 1
        
        # surv_prob at end of interval t
        surv_prob = torch.cumprod(p_surv_interval, dim=1)
        
        # S_prev for interval t is surv_prob[t-1] (or 1 if t=0)
        # Shift surv_prob right and prepend 1s
        ones = torch.ones(batch_size, 1, device=theta.device)
        S_prev = torch.cat([ones, surv_prob[:, :-1]], dim=1) # [batch, T]
        
        # Incidence for each k
        cifs = []
        for k in range(1, K+1):
            p_cond_k = probs[:, :, k]
            incidence_k = S_prev * p_cond_k
            cif_k = torch.cumsum(incidence_k, dim=1)
            cifs.append(cif_k)
            
        # Stack [batch, T, K]
        cifs = torch.stack(cifs, dim=2)
        
        return cifs.detach().cpu().numpy()
