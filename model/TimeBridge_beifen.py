import torch
import torch.nn as nn
from layers.Embed import PatchEmbed
from layers.SelfAttention_Family import TSMixer, ResAttention
from layers.Transformer_EncDec import TSEncoder, IntAttention, PatchSampling, CointAttention
import numpy as np
import torch
import torch.distributions as D

from functorch import vmap, jacfwd, grad
from torch.autograd.functional import jacobian

#nsts
class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self,
                 f_in,
                 f_out,
                 var_num,
                 hidden_dim=128,
                 hidden_layers=2,
                 is_bn=False,
                 dropout=0.05,
                 activation='tanh'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.var_num = var_num
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'ide':
            self.activation = nn.Identity()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError
        if self.hidden_layers == 1:
            self.layers = nn.Sequential(nn.Linear(self.f_in, self.f_out))
        else:
            layers = [nn.Linear(self.f_in, self.hidden_dim),

                      self.activation,
                      nn.Dropout(self.dropout)
                      ]

            for i in range(self.hidden_layers - 2):
                layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                           self.activation,
                           nn.Dropout(dropout)
                           ]
            if is_bn:
                layers += [nn.BatchNorm1d(num_features=self.var_num), nn.Linear(hidden_dim, f_out)]
            else:
                layers += [nn.Linear(hidden_dim, f_out)]
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y


class MLP2(nn.Module):
    """A simple MLP with ReLU activations"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, leaky_relu_slope=0.2):
        super().__init__()
        layers = []
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MyHMM(nn.Module):
    def __init__(self, n_class, lags, x_dim, hidden_dim, mode="mle_scaled:H", num_layers=3) -> None:
        super().__init__()
        self.mode, self.feat = mode.split(":")

        self.initial_prob = torch.nn.Parameter(torch.ones(n_class) / n_class, requires_grad=True)
        self.transition_matrix = torch.nn.Parameter(torch.ones(n_class, n_class) / n_class, requires_grad=True)
        self.observation_means = torch.nn.Parameter(torch.rand(n_class, x_dim), requires_grad=True)
        mask = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]])

        self.mask = torch.tensor(mask)
        # self.mask = torch.tensor(np.array([[1.,1.,1.,1.,0.,0.,],
        #                          [1.,1.,1.,1.,0.,0.,],
        #                            1.,1.,1.,1.,0.,0. ])).cuda()
        # self.observation_means = self.observation_means * self.mask
        self.observation_stddevs = torch.nn.Parameter(torch.rand(n_class, x_dim), requires_grad=True)

        if self.mode == "em":
            self.register_buffer('log_A', torch.randn(n_class, n_class))
            self.register_buffer('log_pi', torch.randn(n_class))
        elif self.mode == "mle_scaled" or self.mode == "mle":
            self.log_A = nn.Parameter(torch.randn(n_class, n_class))

            self.log_pi = nn.Parameter(torch.randn(n_class))
        else:
            raise ValueError("mode must be em or mle_scaled or mle, but got {}".format(self.mode))
        self.n_class = n_class
        self.x_dim = x_dim
        self.lags = lags
        if self.feat == "Ht":
            self.trans = MLP2(input_dim=(lags + 1) * x_dim, hidden_dim=hidden_dim,
                              output_dim=n_class * 2 * x_dim, num_layers=num_layers)
        elif self.feat == "H":
            self.trans = MLP2(input_dim=(1) * x_dim, hidden_dim=hidden_dim,
                              output_dim=n_class * 2 * x_dim, num_layers=num_layers)
        else:
            raise ValueError("feat must be Ht or H")

    def forward_log(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        # length = lags_and_length - self.lags
        log_alpha = torch.zeros(batch_size, length, self.n_class, device=logp_x_c.device)
        log_A = torch.log_softmax(self.log_A, dim=1)
        log_pi = torch.log_softmax(self.log_pi, dim=0)
        for t in range(length):
            if t == 0:
                log_alpha_t = logp_x_c[:, t] + log_pi
            else:
                log_alpha_t = logp_x_c[:, t] + torch.logsumexp(
                    log_alpha[:, t - 1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1)
            log_alpha[:, t] = log_alpha_t
        logp_x = torch.logsumexp(log_alpha[:, -1], dim=-1)
        # logp_x = torch.sum(log_scalers, dim=-1)
        return logp_x

    def forward_backward_log(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        # length = lags_and_length - self.lags
        log_alpha = torch.zeros(batch_size, length, self.n_class, device=logp_x_c.device)
        log_beta = torch.zeros(batch_size, length, self.n_class, device=logp_x_c.device)
        log_scalers = torch.zeros(batch_size, length, device=logp_x_c.device)
        log_A = torch.log_softmax(self.log_A, dim=1)
        log_pi = torch.log_softmax(self.log_pi, dim=0)
        for t in range(length):
            if t == 0:
                log_alpha_t = logp_x_c[:, t] + log_pi
            else:
                log_alpha_t = logp_x_c[:, t] + torch.logsumexp(
                    log_alpha[:, t - 1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1)
            log_scalers[:, t] = torch.logsumexp(log_alpha_t, dim=-1)
            log_alpha[:, t] = log_alpha_t - log_scalers[:, t].unsqueeze(-1)
        log_beta[:, -1] = torch.zeros(batch_size, self.n_class, device=logp_x_c.device)
        for t in range(length - 2, -1, -1):
            log_beta_t = torch.logsumexp(
                log_beta[:, t + 1].unsqueeze(-1) + log_A.unsqueeze(0) + logp_x_c[:, t + 1].unsqueeze(1), dim=-1)
            log_beta[:, t] = log_beta_t - log_scalers[:, t].unsqueeze(-1)
        log_gamma = log_alpha + log_beta
        # logp_x = torch.logsumexp(log_alpha[:, -1],dim=-1)
        logp_x = torch.sum(log_scalers, dim=-1)
        return log_alpha, log_beta, log_scalers, log_gamma, logp_x

    def viterbi_algm(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        log_delta = torch.zeros(batch_size, length, self.n_class, device=logp_x_c.device)
        psi = torch.zeros(batch_size, length, self.n_class, dtype=torch.long, device=logp_x_c.device)

        log_A = torch.log_softmax(self.log_A, dim=1)
        log_pi = torch.log_softmax(self.log_pi, dim=0)
        # log_A = torch.log_softmax(self.log_A,dim=1)
        # log_pi = torch.log_softmax(self.log_pi,dim=0)
        for t in range(length):
            if t == 0:
                log_delta[:, t] = logp_x_c[:, t] + log_pi
            else:
                max_val, max_arg = torch.max(
                    log_delta[:, t - 1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1)
                log_delta[:, t] = max_val + logp_x_c[:, t]
                psi[:, t] = max_arg
        # logp_x = torch.max(log_delta[:, -1])
        c = torch.zeros(batch_size, length, dtype=torch.long, device=logp_x_c.device)
        c[:, -1] = torch.argmax(log_delta[:, -1], dim=-1)
        for t in range(length - 2, -1, -1):
            c[:, t] = psi[:, t + 1].gather(1, c[:, t + 1].unsqueeze(1)).squeeze()
        return c  # , logp_x

    def forward(self, x):
        batch_size, lags_and_length, _ = x.shape
        length = lags_and_length - self.lags
        # x_H = (batch_size, length, (lags) * x_dim)
        # x_H = x.unfold(dimension=1, size=self.lags+1, step=1).transpose(-2, -1)  #  256 x 6 x 2 x 4
        # if self.feat == "H":
        #     x_H = x_H[...,:self.lags,:].reshape(batch_size, length, -1)
        # elif self.feat == "Ht":
        #     x_H = x_H.reshape(batch_size, length, -1)
        x_H = x
        # (batch_size, length, n_class, x_dim)
        # out = self.trans(x_H).reshape(batch_size, length, self.n_class, 2 * self.x_dim)
        # mus, logvars = out[..., :self.x_dim], out[..., self.x_dim:] # batch x length x n_class x x_dim
        # dist = tD.Normal(mus, torch.exp(logvars / 2))
        dist = D.Normal(self.observation_means[:, :4], torch.relu(self.observation_stddevs[:, :4]) + 1e-1)
        # [B, L, C, 4]
        # print(x[:, self.lags:].unsqueeze(2).shape)
        # exit()
        # logp_x_c = dist.log_prob(x[:, self.lags:].unsqueeze(2)).sum(-1)  # (batch_size, length, n_class)
        logp_x_c = dist.log_prob(x[:, :, :4].unsqueeze(2)).sum(-1) # [B , L, C]



        if self.mode == "em" or self.mode == "mle_scaled":
            log_alpha, log_beta, log_scalers, log_gamma, logp_x = self.forward_backward_log(logp_x_c)
            if self.mode == "em":
                batch_normalizing_factor = torch.log(torch.tensor(batch_size, device=logp_x_c.device))
                expected_log_pi = log_gamma[:, 0, :] - log_gamma[:, 0, :].logsumexp(dim=-1).unsqueeze(-1)
                expected_log_pi = expected_log_pi.logsumexp(dim=0) - batch_normalizing_factor
                log_A = torch.log_softmax(self.log_A, dim=1)
                log_xi = torch.zeros(batch_size, length - 1, self.n_class, self.n_class, device=logp_x_c.device)
                for t in range(length - 1):  # B,Ct,1 B,1,Ct+1 1,Ct,Ct+1 B,1,Ct+1,
                    log_xi_t = log_alpha[:, t].unsqueeze(-1) + log_beta[:, t + 1].unsqueeze(1) + log_A.unsqueeze(
                        0) + logp_x_c[:, t + 1].unsqueeze(1)
                    log_xi_scalers = torch.logsumexp(log_xi_t, dim=(1, 2), keepdim=True)
                    log_xi[:, t] = log_xi_t - log_xi_scalers
                expected_log_A = torch.logsumexp(log_xi, dim=1) - torch.logsumexp(log_xi, dim=(1, 3)).unsqueeze(-1)
                expected_log_A = expected_log_A.logsumexp(dim=0) - batch_normalizing_factor
                self.log_A = expected_log_A.detach()
                self.log_pi = expected_log_pi.detach()
        elif self.mode == "mle":
            logp_x = self.forward_log(logp_x_c)

        c_est = self.viterbi_algm(logp_x_c)

        return logp_x, c_est

    pass


class Encoder_ZD(nn.Module):
    def __init__(self, configs) -> None:
        super(Encoder_ZD, self).__init__()
        self.configs = configs
        # self.zd_net = nn.MultiheadAttention(embed_dim=self.configs.dynamic_dim, num_heads=self.configs.n_heads)
        self.zd_net = nn.Linear(in_features=self.configs.enc_in, out_features=self.configs.zd_dim)

        self.enc_embedding = nn.Sequential(
            MLP(configs.seq_len, configs.dynamic_dim, var_num=self.configs.enc_in,
                activation=self.configs.activation,
                hidden_dim=configs.hidden_dim,
                hidden_layers=configs.hidden_layers, dropout=configs.dropout, is_bn=self.configs.is_bn)
        )
        self.zd_pred_net_mean = MLP(configs.seq_len, configs.pred_len, var_num=self.configs.enc_in,
                                    activation=self.configs.activation,
                                    hidden_dim=configs.hidden_dim,
                                    hidden_layers=configs.hidden_layers, dropout=configs.dropout)
        self.zd_pred_net_std = MLP(configs.seq_len, configs.pred_len, var_num=self.configs.enc_in,
                                   activation=self.configs.activation,
                                   hidden_dim=configs.hidden_dim,
                                   hidden_layers=configs.hidden_layers, dropout=configs.dropout)
        self.nonstationary_transition_prior = NPChangeTransitionPrior(lags=0,
                                                                      latent_size=self.configs.zd_dim,
                                                                      embedding_dim=self.configs.embedding_dim,
                                                                      num_layers=1,
                                                                      hidden_dim=self.configs.hidden_dim)
        # self.nonstationary_transition_prior = NPChangeTransitionPrior(lags=0,
        #                                                               latent_size=self.configs.zd_dim,
        #                                                               embedding_dim=self.configs.embedding_dim,
        #                                                               num_layers=1,
        #                                                               hidden_dim=3)
        self.zd_rec_net_mean = nn.Sequential(
            MLP(configs.dynamic_dim, configs.seq_len, var_num=self.configs.enc_in, activation=self.configs.activation,
                hidden_dim=configs.hidden_dim,
                hidden_layers=configs.hidden_layers, dropout=configs.dropout)
        )
        self.zd_rec_net_std = nn.Sequential(
            MLP(configs.dynamic_dim, configs.seq_len, var_num=self.configs.enc_in, activation=self.configs.activation,
                hidden_dim=configs.hidden_dim,
                hidden_layers=configs.hidden_layers, dropout=configs.dropout)
        )
        self.register_buffer('nonstationary_dist_mean', torch.zeros(self.configs.zd_dim))
        self.register_buffer('nonstationary_dist_var', torch.eye(self.configs.zd_dim))

    @property
    def nonstationary_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.nonstationary_dist_mean, self.nonstationary_dist_var)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x_enc):
        zd_x_enc = self.enc_embedding(x_enc.permute(0, 2, 1))
        # zd = self.zd_net(zd_x_enc, zd_x_enc, zd_x_enc)[0][:, :self.configs.zd_dim]
        zd = self.zd_net(zd_x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        zd_rec_mean = self.zd_rec_net_mean(zd)
        zd_rec_std = self.zd_rec_net_std(zd)
        zd_rec = self.reparametrize(zd_rec_mean, zd_rec_std)
        zd_pred_mean = self.zd_pred_net_mean(zd_rec_mean)
        zd_pred_std = self.zd_pred_net_std(zd_rec_mean)
        zd_pred = self.reparametrize(zd_pred_mean, zd_pred_std)
        return (zd_rec_mean, zd_rec_std, zd_rec), (zd_pred_mean, zd_pred_std, zd_pred)

    def kl_loss(self, mus, logvars, z_est, c_embedding):
        lags_and_length = z_est.shape[1]

        # 0: 添加方差下界正则化，防止后验坍缩
        # 计算 logvars 的平均值，如果太小，添加惩罚
        logvars_mean = logvars.mean()
        variance_penalty = torch.relu(-2.0 - logvars_mean) * 0.1  # 如果平均 logvar < -2，添加惩罚

        # 1: 限制 logvars 范围，防止方差过小或过大导致数值不稳定
        logvars = torch.clamp(logvars, min=-5.0, max=10.0)  # 改为 -5.0，不要让方差太小
        # 处理 NaN
        logvars = torch.nan_to_num(logvars, nan=0.0, posinf=10.0, neginf=-5.0)

        # 2: 添加 epsilon 防止数值下溢
        std = torch.exp(logvars / 2) + 1e-6
        q_dist = D.Normal(mus, std)
        log_qz = q_dist.log_prob(z_est)

        # 检查是否有 NaN 或 Inf
        # if torch.isnan(log_qz).any() or torch.isinf(log_qz).any():
        #     print(f"⚠️ Warning: NaN or Inf in log_qz! min={log_qz.min():.2e}, max={log_qz.max():.2e}")
        #     log_qz = torch.nan_to_num(log_qz, nan=0.0, posinf=100.0, neginf=-100.0)

        # Future KLD
        log_qz_laplace = log_qz
        residuals, logabsdet = self.nonstationary_transition_prior.forward(z_est, c_embedding)

        # 3: 限制 residuals 范围，防止 log_prob 计算出极值
        residuals = torch.clamp(residuals, min=-50.0, max=50.0)

        # 3.5: 限制 logabsdet 范围
        logabsdet = torch.clamp(logabsdet, min=-100.0, max=100.0)

        log_pz_laplace = torch.sum(self.nonstationary_dist.log_prob(
            residuals), dim=1) + logabsdet.sum(dim=1)

        # 调试信息
        # if torch.rand(1).item() < 0.02:  # 2% 概率打印
        #     log_qz_sum = torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1).mean()
        #     print(f"[ZD KL Debug] log_q_sum: {log_qz_sum:.2e}, log_p: {log_pz_laplace.mean():.2e}, "
        #           f"logvar_mean: {logvars_mean:.2e}")

        kld_laplace = (
                              torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (
                          lags_and_length)
        kld_laplace = kld_laplace.mean()

        # 4: 使用绝对值 + 方差惩罚
        loss = torch.abs(kld_laplace) + variance_penalty

        # 记录原始 KL 的符号
        if torch.rand(1).item() < 0.02:  # 2% 概率打印
            if kld_laplace < -5.0:
                print(f"[ZD] Negative KL: {kld_laplace:.4f} -> abs: {torch.abs(kld_laplace):.4f}, "
                      f"var_penalty: {variance_penalty:.4f}")

        return loss


class Encoder_ZC(nn.Module):
    def __init__(self, configs) -> None:
        super(Encoder_ZC, self).__init__()
        self.configs = configs
        # latent_size 是啥来的，#HMM跟先验的lags是一个东西吗
        if configs.enc_in < 100:
            self.stationary_transition_prior = NPTransitionPrior(lags=self.configs.lags,
                                                                 latent_size=self.configs.zc_dim,
                                                                 num_layers=1,
                                                                 hidden_dim=self.configs.hidden_dim)
        else:
            self.stationary_transition_prior = NPTransitionPrior(lags=self.configs.lags,
                                                                 latent_size=self.configs.zc_dim,
                                                                 num_layers=1,
                                                                 hidden_dim=3)

        self.zc_rec_net_mean = nn.Sequential(
            MLP(configs.seq_len, configs.seq_len, var_num=self.configs.zc_dim, activation=self.configs.activation,
                hidden_dim=configs.hidden_dim,
                hidden_layers=configs.hidden_layers, dropout=configs.dropout, is_bn=self.configs.is_bn)
        )

        self.zc_rec_net_std = nn.Sequential(
            MLP(configs.seq_len, configs.seq_len, var_num=self.configs.zc_dim, activation=self.configs.activation,
                hidden_dim=configs.hidden_dim,
                hidden_layers=configs.hidden_layers, dropout=configs.dropout, is_bn=self.configs.is_bn)
        )
        self.zc_pred_net_mean = MLP(configs.seq_len, configs.pred_len, var_num=self.configs.zc_dim,
                                    activation=self.configs.activation,
                                    hidden_dim=configs.hidden_dim,
                                    hidden_layers=configs.hidden_layers, dropout=configs.dropout,
                                    is_bn=self.configs.is_bn)

        self.zc_pred_net_std = MLP(configs.seq_len, configs.pred_len, var_num=self.configs.zc_dim,
                                   activation=self.configs.activation,
                                   hidden_dim=configs.hidden_dim,
                                   hidden_layers=configs.hidden_layers, dropout=configs.dropout,
                                   is_bn=self.configs.is_bn)

        self.zc_kl_weight = configs.zc_kl_weight
        self.lags = self.configs.lags
        self.register_buffer('stationary_dist_mean', torch.zeros(self.configs.zc_dim))
        self.register_buffer('stationary_dist_var', torch.eye(self.configs.zc_dim))

    @property
    def stationary_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.stationary_dist_mean, self.stationary_dist_var)

    def forward(self, x_enc):
        zc_rec_mean = self.zc_rec_net_mean(x_enc.permute(0, 2, 1))
        zc_rec_std = self.zc_rec_net_std(x_enc.permute(0, 2, 1))
        zc_rec = self.reparametrize(zc_rec_mean, zc_rec_std)
        zc_pred_mean = self.zc_pred_net_mean(zc_rec_mean)
        zc_pred_std = self.zc_pred_net_std(zc_rec_mean)
        zc_pred = self.reparametrize(zc_pred_mean, zc_pred_std)

        return (zc_rec_mean, zc_rec_std, zc_rec), (zc_pred_mean, zc_pred_std, zc_pred)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def kl_loss(self, mus, logvars, z_est):
        lags_and_length = z_est.shape[1]

        # 0: 添加方差下界正则化
        # logvars_mean = logvars.mean()
        # variance_penalty = torch.relu(-2.0 - logvars_mean) * 0.1

        # 1: 限制 logvars 范围，不要太小
        logvars = torch.clamp(logvars, min=-5.0, max=10.0)
        # 处理 NaN
        logvars = torch.nan_to_num(logvars, nan=0.0, posinf=10.0, neginf=-5.0)

        # 2: 添加 epsilon 防止数值下溢
        std = torch.exp(logvars / 2) + 1e-6
        q_dist = D.Normal(mus, std)
        log_qz = q_dist.log_prob(z_est)

        # 检查是否有 NaN 或 Inf
        # if torch.isnan(log_qz).any() or torch.isinf(log_qz).any():
        #     print(f"⚠️ Warning: NaN or Inf in log_qz (ZC)! min={log_qz.min():.2e}, max={log_qz.max():.2e}")
        #     log_qz = torch.nan_to_num(log_qz, nan=0.0, posinf=100.0, neginf=-100.0)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()

        # Future KLD
        log_qz_laplace = log_qz[:, self.lags:]
        residuals, logabsdet = self.stationary_transition_prior(z_est)

        # # 3: 限制 residuals 范围
        # residuals = torch.clamp(residuals, min=-50.0, max=50.0)
        #
        # # 3.5: 限制 logabsdet 范围
        # logabsdet = torch.clamp(logabsdet, min=-100.0, max=100.0)

        log_pz_laplace = torch.sum(self.stationary_dist.log_prob(
            residuals), dim=1) + logabsdet.sum(dim=1)

        # 调试信息
        # if torch.rand(1).item() < 0.02:  # 2% 概率打印
        #     print(f"[ZC KL Debug] kld_normal: {kld_normal:.2e}, kld_laplace: {(torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace).mean()/(lags_and_length - self.lags):.2e}, "
        #           f"logvar_mean: {logvars_mean:.2e}")

        kld_laplace = (
                              torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (
                              lags_and_length - self.lags)
        kld_laplace = kld_laplace.mean()

        # 4: 使用绝对值 + 方差惩罚
        kld_normal_abs = torch.abs(kld_normal)
        kld_laplace_abs = torch.abs(kld_laplace)

        loss = kld_normal_abs + kld_laplace_abs


        return loss


class Decoder(nn.Module):
    def __init__(self, configs) -> None:
        super(Decoder, self).__init__()
        self.configs = configs
        self.z_net = nn.Linear(self.configs.zd_dim + self.configs.enc_in, self.configs.enc_in, bias=False)
        self.pred_net = MLP(configs.pred_len, configs.pred_len, var_num=self.configs.enc_in,
                            hidden_dim=configs.hidden_dim,
                            hidden_layers=configs.hidden_layers, is_bn=self.configs.is_bn)

        self.rec_net = MLP(configs.seq_len, configs.seq_len, var_num=self.configs.enc_in,
                           hidden_dim=configs.hidden_dim,
                           hidden_layers=configs.hidden_layers, is_bn=self.configs.is_bn)

        weight = torch.eye(configs.enc_in, self.configs.zd_dim + self.configs.enc_in)

        self.z_net.weight = nn.Parameter(weight)

    def forward(self, zc_rec, zd_rec, zc_pred, zd_pred):
        z_rec = self.z_net(torch.cat([zc_rec, zd_rec], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
        z_pred = self.z_net(torch.cat([zc_pred, zd_pred], dim=1).permute(0, 2, 1)).permute(0, 2, 1)

        x = self.rec_net(z_rec).permute(0, 2, 1)
        y = self.pred_net(z_pred).permute(0, 2, 1)

        return x, y


class NPTransitionPrior(nn.Module):

    def __init__(self, lags, latent_size, num_layers=3, hidden_dim=64, compress_dim=10):
        super().__init__()
        self.lags = lags
        self.latent_size = latent_size
        # 注意: latent_size 这里实际上是 x_dim (原始特征维度), 不是隐变量维度
        # 当 latent_size > 100 时, forward 中会先压缩 lags*latent_size 到 compress_dim
        # 所以 input_dim 应该是 compress_dim + 1
        # 当 latent_size <= 100 时, forward 中直接使用 lags*latent_size
        # 所以 input_dim 应该是 lags*latent_size + 1
        self.gs = nn.ModuleList([MLP2(input_dim=compress_dim + 1, hidden_dim=hidden_dim,
                                      output_dim=1, num_layers=num_layers) for _ in
                                 range(latent_size)]) if latent_size > 100 else nn.ModuleList(
            [MLP2(input_dim=lags * latent_size + 1, hidden_dim=hidden_dim,
                  output_dim=1, num_layers=num_layers) for _ in range(latent_size)])

        self.compress = nn.Linear(lags * latent_size, compress_dim)
        self.compress_dim = compress_dim
        # self.fc = MLP(input_dim=embedding_dim,hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=2)

    def forward(self, x, mask=None):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags
        # batch_x: (batch_size, lags+length, x_dim) -> (batch_size, length, lags+1, x_dim)
        batch_x = x.unfold(dimension=1, size=self.lags +
                                             1, step=1).transpose(2, 3)
        batch_x = batch_x.reshape(-1, self.lags + 1, x_dim)
        batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim) 历史数据窗口
        batch_x_t = batch_x[:, -1]  # (batch_size*length, x_dim) 最后一个时间点数据

        # (batch_size*length, lags*x_dim)
        batch_x_lags = batch_x_lags.reshape(-1, self.lags * x_dim)
        if x.shape[-1] > 100:
            batch_x_lags = self.compress(batch_x_lags)
        sum_log_abs_det_jacobian = 0
        residuals = []
        for i in range(self.latent_size):
            # (batch_size x length, hidden_dim + lags*x_dim + 1)

            if mask is not None:
                batch_inputs = torch.cat(
                    (batch_x_lags * mask[i], batch_x_t[:, i:i + 1]), dim=-1)
            else:
                batch_inputs = torch.cat(
                    (batch_x_lags, batch_x_t[:, i:i + 1]), dim=-1)

            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)

            J = jacfwd(self.gs[i])
            data_J = vmap(J)(batch_inputs).squeeze()
            logabsdet = torch.log(torch.abs(data_J[:, -1]))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)
        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)

        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian


class NPChangeTransitionPrior(nn.Module):

    def __init__(
            self,
            lags,
            latent_size,
            embedding_dim,
            num_layers=3,
            hidden_dim=64):
        super().__init__()
        self.latent_size = latent_size
        self.lags = lags
        self.gs = nn.ModuleList([MLP2(input_dim=embedding_dim + 1, hidden_dim=hidden_dim,
                                      output_dim=1, num_layers=num_layers) for _ in range(latent_size)])
        self.fc = MLP2(input_dim=embedding_dim, hidden_dim=hidden_dim,
                       output_dim=hidden_dim, num_layers=num_layers)

    def forward(self, x, embeddings):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags
        # batch_x: (batch_size, lags+length, x_dim) -> (batch_size, length, lags+1, x_dim)
        batch_x = x.unfold(dimension=1, size=self.lags +
                                             1, step=1).transpose(2, 3)
        # (batch_size, lags+length, hidden_dim)
        # embeddings = self.fc(embeddings)
        # batch_embeddings: (batch_size, lags+length, hidden_dim) -> (batch_size, length, lags+1, hidden_dim) -> (batch_size*length, hidden_dim)
        # batch_embeddings = embeddings.unfold(
        #     dimension=1, size=self.lags+1, step=1).transpose(2, 3)[:, :, -1].reshape(batch_size * length, -1)
        batch_embeddings = embeddings[:, -length:].expand(batch_size, length, -1).reshape(batch_size * length, -1)
        batch_x = batch_x.reshape(-1, self.lags + 1, x_dim)
        batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim)
        batch_x_t = batch_x[:, -1:]  # (batch_size*length, x_dim)
        # (batch_size*length, lags*x_dim)
        # batch_x_lags = batch_x_lags.reshape(-1, self.lags * x_dim)
        sum_log_abs_det_jacobian = 0
        residuals = []
        for i in range(self.latent_size):
            # (batch_size x length, hidden_dim + lags*x_dim + 1)

            batch_inputs = torch.cat(
                (batch_embeddings, batch_x_t[:, :, i]), dim=-1)
            # 并行不了

            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)

            J = jacfwd(self.gs[i])
            data_J = vmap(J)(batch_inputs).squeeze()
            logabsdet = torch.log(torch.abs(data_J[:, -1]))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
    # nsts_o部分初始化
        self.configs = configs
        self.zc_dim = int(self.configs.enc_in*0.7)
        self.zd_dim = self.configs.enc_in - self.zc_dim
    # 加入新的参数维度
        self.configs.zc_dim = self.zc_dim
        self.configs.zd_dim = self.zd_dim

        self.encoder_zd = Encoder_ZD(configs)
        self.encoder_zc = Encoder_ZC(configs)
        self.decoder = Decoder(configs)
        self.encoder_u = MyHMM(n_class=self.configs.n_class, lags=0,
                               x_dim=self.configs.enc_in, hidden_dim=self.configs.hidden_dim, mode="mle_scaled:H")
        self.c_embeddings = nn.Embedding(configs.n_class, configs.embedding_dim)

        self.rec_criterion = nn.MSELoss()
        self.revin = configs.revin  # long-term with temporal

        self.c_in = configs.enc_in
        self.period = configs.period
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_p = self.seq_len // self.period
        if configs.num_p is None:
            configs.num_p = self.num_p

        self.embedding = PatchEmbed(configs, num_p=self.num_p)
        self.embedding1 = PatchEmbed(configs, num_p=self.num_p)

        layers = self.layers_init(configs)
        self.encoder = TSEncoder(layers)
        self.encoder1 = TSEncoder(layers)

        out_p = self.num_p if configs.pd_layers == 0 else configs.num_p
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)
        )
        self.decoder1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)
        )

        self.decoder_x = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.seq_len, bias=False)
        )
        self.decoder1_x = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.seq_len, bias=False)
        )

        self.final_mlp_x = nn.Sequential(
            nn.Linear(self.c_in, self.c_in),
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(self.c_in, self.c_in),
        )
        with torch.no_grad():
            # 直接使用torch.eye创建单位矩阵
            self.final_mlp[0].weight.data = torch.eye(self.c_in)

            # 将偏置初始化为零
            self.final_mlp[0].bias.data.zero_()

    def layers_init(self, configs):
        integrated_attention = [IntAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout, stable_len=configs.stable_len,
            activation=configs.activation, stable=True, enc_in=self.c_in
        ) for i in range(configs.ia_layers)]

        patch_sampling = [PatchSampling(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, stable=False, stable_len=configs.stable_len,
            in_p=self.num_p if i == 0 else configs.num_p, out_p=configs.num_p,
            dropout=configs.dropout, activation=configs.activation
        ) for i in range(configs.pd_layers)]

        cointegrated_attention = [CointAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout),
                    configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout,
            activation=configs.activation, stable=False, enc_in=self.c_in, stable_len=configs.stable_len,
        ) for i in range(configs.ca_layers)]

        return [*integrated_attention, *patch_sampling, *cointegrated_attention]

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,y_enc=None,is_train=True, is_out_u=False, c_est=None):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros((*x_enc.shape[:-1], 4), device=x_enc.device)

        other_loss = torch.tensor(0.0, device=x_enc.device)

        mean, std = (x_enc.mean(1, keepdim=True).detach(),
                     x_enc.std(1, keepdim=True).detach())
        x_enc = (x_enc - mean) / (std + 1e-5)

        x_mean = self.embedding(x_enc, x_mark_enc)
        x_mean = self.encoder(x_mean)[0][:, :self.c_in, ...]
        BXD_mean = self.decoder_x(x_mean).transpose(-1, -2) # [B,X,D]
        BYD_mean = self.decoder(x_mean).transpose(-1, -2)  # [B,Y,D]

        x_std = self.embedding1(x_enc, x_mark_enc)
        x_std = self.encoder1(x_std)[0][:, :self.c_in, ...]
        BXD_std = self.decoder1_x(x_std).transpose(-1, -2)  # [B,X,D]
        BYD_std = self.decoder1(x_std).transpose(-1, -2)  # [B,Y,D]

        dec_out = self.reparametrize(BYD_mean, BYD_std)
        # BYD_mean,BYD_std,dec_out  shape [B,pred_len,D]
        zc_pred_mean, zd_pred_mean = torch.split(BYD_mean, [self.zc_dim, self.zd_dim], dim=2)
        zc_pred_std, zd_pred_std = torch.split(BYD_std, [self.zc_dim, self.zd_dim], dim=2)
        zc_pred, zd_pred = torch.split(dec_out, [self.zc_dim, self.zd_dim], dim=2)
        # print('zc_pred_mean', zc_pred_mean.shape)
        # print('zc_pred_std', zc_pred_std.shape)
        # print('zc_pred', zc_pred.shape)

        dec_out_x = self.reparametrize(BXD_mean, BXD_std)
        # BXD_mean,BXD_mean,dec_out_x  shape [B,seq_len,D]
        zc_rec_mean, zd_rec_mean = torch.split(BXD_mean, [self.zc_dim, self.zd_dim], dim=2)
        zc_rec_std, zd_rec_std = torch.split(BXD_mean, [self.zc_dim, self.zd_dim], dim=2)
        zc_rec, zd_rec = torch.split(dec_out_x, [self.zc_dim, self.zd_dim], dim=2)

        # print('zc_rec_mean', zc_rec_mean.shape)
        # print('zc_rec_std', zc_rec_std.shape)
        # print('zc_rec', zc_rec.shape)

        #print(torch.cat([zc_rec_mean.permute(0, 2, 1), zc_pred_mean.permute(0, 2, 1)], dim=2).permute(0, 2, 1).shape)

        # dec_out = self.final_mlp(dec_out)
        x = self.final_mlp_x(dec_out_x)
        y = dec_out * std + mean


        if is_train and (not self.configs.No_prior):
            y_enc = (y_enc - mean) / (std + 1e-5)
            hmm_loss = 0
            if c_est == None:
                E_logp_x, c_est = self.encoder_u(torch.cat([x_enc, y_enc], dim=1))
                hmm_loss = -E_logp_x.mean()
            embeddings = self.c_embeddings(c_est)

            # 正确的拼接方式：在时间维度(dim=1)上拼接 rec 和 pred
            # zc_rec_mean: (B, seq_len, zc_dim), zc_pred_mean: (B, pred_len, zc_dim)
            # cat后: (B, seq_len+pred_len, zc_dim)
            zc_kl_loss = self.encoder_zc.kl_loss(torch.cat([zc_rec_mean, zc_pred_mean], dim=1),
                                                 torch.cat([zc_rec_std, zc_pred_std], dim=1),
                                                 torch.cat([zc_rec, zc_pred], dim=1))
            zd_kl_loss = self.encoder_zd.kl_loss(torch.cat([zd_rec_mean, zd_pred_mean], dim=1),
                                                 torch.cat([zd_rec_std, zd_pred_std], dim=1),
                                                 torch.cat([zd_rec, zd_pred], dim=1), embeddings)
            other_loss = zc_kl_loss * self.configs.zc_kl_weight + zd_kl_loss * self.configs.zd_kl_weight + hmm_loss * self.configs.hmm_weight
            # print("zc_kl_loss:", zc_kl_loss.item(), "zd_kl_loss:", zd_kl_loss.item(), "hmm_loss:", hmm_loss.item(), "other_loss:", other_loss.item())
            if is_out_u:
                return y, x, other_loss, c_est
        return y, x, other_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, y_enc=None, is_train=True, is_out_u=False, c_est=None):

        # 将所有参数正确传递给 forecast
        forecast_results = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec,
            y_enc=y_enc, is_train=is_train, is_out_u=is_out_u, c_est=c_est
        )

        # 根据 is_out_u 处理不同的返回值数量
        if is_out_u:
            # 训练时（is_out_u=True），forecast 返回 4 个值
            y, x, other_loss, U = forecast_results
            return y[:, -self.pred_len:, :], x, other_loss, U
        else:
            # 验证/测试时（is_out_u=False），forecast 返回 3 个值
            y, x, other_loss = forecast_results
            return y[:, -self.pred_len:, :], x, other_loss
