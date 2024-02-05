import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#precalculated variables

beta = torch.linspace(1e-4, 0.02, 1000)
beta = beta.to(device)

alpha = 1. - beta
alpha = alpha.to(device)

alpha_hat = torch.cumprod(alpha, dim = 0)
alpha_hat = alpha_hat.to(device)
alpha_hat_prev = nn.functional.pad(alpha_hat[:-1], (1, 0), value=1.0)
sqrt_recip_alpha = torch.sqrt(1.0 / alpha)

alpha_hat_sqrt = torch.sqrt(alpha_hat)
alpha_hat_sqrt = alpha_hat_sqrt.to(device)
one_minus_alpha_hat = 1. - alpha_hat
one_minus_alpha_hat_sqrt = torch.sqrt(one_minus_alpha_hat)
one_minus_alpha_hat_sqrt = one_minus_alpha_hat_sqrt.to(device)
posterior_variance = beta * (1. - alpha_hat_prev) / (1. - alpha_hat)
posterior_variance = posterior_variance.to(device)