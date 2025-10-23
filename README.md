# A-MMMSE
This represitory is an implementation of the paper: An Accelerated Mixed Weighted-Unweighted MMSE Approach for MU-MIMO Beamforming.
## Introduction
The weighted sum-rate (WSR) maximization problem plays a central role in precoding design for downlink multi-user multiple-input multiple-output (MU-MIMO) systems. We consider a single-cell MU-MIMO downlink, where a base station (BS) equipped with $M$ transmit antennas serves $K$ users, each with $N$ receive antennas, by simultaneously transmitting $d$ independent data streams. Let $\mathbf{s}_k \in \mathbb{C}^{d \times 1}$ denote the symbol vector intended for user $k$, and $\mathbf{V}_k \in \mathbb{C}^{M \times d}$ be the corresponding linear precoder. The received signal at user $k$ is given by:

$$
\mathbf{y}_{k} = \mathbf{H}_{k} \mathbf{V}_{k} \mathbf{s}_{k} + \sum_{\substack{j=1 \\ j \neq k}}^{K} \mathbf{H}_{k} \mathbf{V}_{j} \mathbf{s}_{j} + \mathbf{n}_{k},
$$

where $\mathbf{H}_k \in \mathbb{C}^{N \times M}$ is the channel matrix from the BS to user $k$, and $\mathbf{n}_k \in \mathbb{C}^{N \times 1}$ is the additive white Gaussian noise vector following $\mathcal{CN}(\mathbf{0}, \sigma_k^2 \mathbf{I})$.
The WSR maximization problem over the set of precoders $\mathbf{V} \triangleq {\mathbf{V}_{k}}_{k=1}^K$ is formulated as:


$$
    \begin{aligned}
        &\underset{\mathbf{V}}{\text{max}} && \sum_{k=1}^{K} \alpha_{k} R_{k} \\
        & \text{s.t.} && \sum_{k=1}^{K} \text{Tr}\left(\mathbf{V}_{k} \mathbf{V}_{k}^{H}\right) \leq P_{\max},
    \end{aligned}
$$

where $\alpha_k$ is the priority weight of user $k$, $P_{\text{max}}$ denotes the total transmit power budget at the BS, and $R_k$ represents the achievable rate for user $k$, defined as:

$$
R_{k} \triangleq \log \text{det} \left(\mathbf{I} + \mathbf{H}_{k} \mathbf{V}_{k} \mathbf{V}_{k}^{H} \mathbf{H}_{k}^{H} \left( \sum_{j \neq k} \mathbf{H}_{k} \mathbf{V}_{j} \mathbf{V}_{j}^{H} \mathbf{H}_{k}^{H} + \sigma_k^{2} \mathbf{I} \right)^{-1} \right).
$$
