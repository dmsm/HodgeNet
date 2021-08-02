import numpy as np
import scipy
import scipy.sparse.linalg
import torch
import torch.multiprocessing


def repeat_d_matrix(d, n_repeats):
    """Create block diagonal d matrix for vectorial operator."""
    if n_repeats == 1:
        return d

    I, J, V = scipy.sparse.find(d)

    bigI = np.concatenate([I*n_repeats+idx for idx in range(n_repeats)])
    bigJ = np.concatenate([J*n_repeats+idx for idx in range(n_repeats)])
    bigV = np.tile(V, n_repeats)

    result = scipy.sparse.csr_matrix((bigV, (bigI, bigJ)),
                                     shape=(d.shape[0]*n_repeats,
                                            d.shape[1]*n_repeats))

    return result


def single_forward(ctx_eigenvalues, ctx_eigenvectors, ctx_dx, star0, star1, d,
                   n_eig, n_extra_eig, device):
    """Compute the eigenvectors and eigenvalues of the generalized eigensystem.

        L*x = lambda*A*x
    where
        L = d'*blockdiag(star1)*d
        A = blockdiag(star0)
    """
    ne, nv = d.shape
    nvec = star1.shape[1]

    # make L
    star1s = [star1[i].squeeze() for i in range(ne)]
    star1mtx = scipy.sparse.block_diag(star1s)
    drep = repeat_d_matrix(d, nvec)
    L = drep.T @ (star1mtx @ drep)

    # make A
    star0s = [star0[i].squeeze() for i in range(nv)]
    star0mtx = scipy.sparse.block_diag(star0s)

    # can compute extra eigenvectors beyond n_eig (k total)
    # extras will improve quality of derivatives
    k = n_eig + n_extra_eig + nvec  # adding nvec because all zero eigenvalues
    shift = 1e-4  # for numerical stability

    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        L + shift*scipy.sparse.eye(L.shape[0]),
        k=k, M=star0mtx, which='LM', sigma=0, tol=1e-3)
    eigenvalues -= shift

    # sort eigenvalues/corresponding eigenvectors and make sign consistent
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors * np.sign(eigenvectors[0])
    eigenvalues[:nvec] = 0  # first Laplacian eigenvalue is always zero
    eigenvectors[:, 0] = 1  # will normalize momentarily

    # normalize eigenvectors
    vec_norms = np.sqrt(
        np.sum(eigenvectors * (star0mtx @ eigenvectors), axis=0))
    eigenvectors = eigenvectors / vec_norms.clip(1e-4)

    reshaped_eigenvectors = np.swapaxes(
        np.reshape(eigenvectors, (nv, nvec, k)), 1, 2)

    # differentiate eigenvectors --- useful for the derivative during backprop
    d_eig = np.swapaxes(np.reshape(drep @ eigenvectors, (ne, nvec, k)), 1, 2)

    ctx_eigenvalues.copy_(torch.from_numpy(eigenvalues).to(device))
    ctx_eigenvectors.copy_(torch.from_numpy(reshaped_eigenvectors).to(device))
    ctx_dx.copy_(torch.from_numpy(d_eig).to(device))


def single_backward(dx, eigenvalues, eigenvectors, n_eig,
                    grad_output_eigenvalues, grad_output_eigenvectors, device):
    """Backward pass for the eigenproblem."""
    nvec = eigenvectors.shape[2]

    grad_output_eigenvalues = torch.cat(
        [torch.zeros(nvec).to(device), grad_output_eigenvalues])
    dstar1 = torch.einsum('i,eil,eim->elm',
                          grad_output_eigenvalues, dx[:, :n_eig],
                          dx[:, :n_eig])
    dstar0 = torch.einsum('i,i,wil,wim->wlm',
                          -grad_output_eigenvalues, eigenvalues[:n_eig],
                          eigenvectors[:, :n_eig], eigenvectors[:, :n_eig])

    grad_output_eigenvectors = torch.cat([
        torch.zeros(eigenvectors.shape[0], nvec, nvec).to(device),
        grad_output_eigenvectors], dim=1)

    total_eig = eigenvectors.shape[1]  # includes the extra eigenvalues

    M = eigenvalues[:, None].repeat(1, total_eig)
    M = 1. / (M - M.t())
    M[np.diag_indices(total_eig)] = 0
    M[:nvec, :nvec] = 0

    dstar1 += torch.einsum('vjn,vin,ij,ejl,eim->elm', eigenvectors,
                           grad_output_eigenvectors, M[:n_eig], dx,
                           dx[:, :n_eig])

    N = eigenvalues[:, None].repeat(1, total_eig)
    N = N / (N.t() - N)
    N[:nvec, :nvec] = 0
    N[np.diag_indices(total_eig)] = -.5

    dstar0 += torch.einsum('vjn,vin,ij,wjl,wim->wlm', eigenvectors,
                           grad_output_eigenvectors, N[:n_eig], eigenvectors,
                           eigenvectors[:, :n_eig])

    return dstar0, dstar1


class HodgeEigensystem(torch.autograd.Function):
    """Autograd class for solving batches of Hodge eigensystems.

    WARNING: Assumes that the Hodge star matrices are symmetric.
    """

    @staticmethod
    def forward(ctx, nb, n_eig, n_extra_eig, *inputs):
        ctx.device = inputs[1].device
        ctx.nb = nb
        ctx.n_eig = n_eig

        eigenvalues = [
            torch.empty(
                inputs[3*i+1].shape[1] + n_eig + n_extra_eig
            ).to(ctx.device).share_memory_()
            for i in range(nb)
        ]
        eigenvectors = [
            torch.empty(
                inputs[3*i+2].shape[1],
                inputs[3*i+1].shape[1] + n_eig + n_extra_eig,
                inputs[3*i+1].shape[1]
            ).to(ctx.device).share_memory_()
            for i in range(nb)
        ]
        dx = [
            torch.empty(
                inputs[3*i+2].shape[0],
                inputs[3*i+1].shape[1] + n_eig + n_extra_eig,
                inputs[3*i+1].shape[1]
            ).to(ctx.device).share_memory_()
            for i in range(nb)
        ]

        processes = []
        for i in range(nb):
            star0, star1, d = inputs[3*i:3*i+3]
            p = torch.multiprocessing.Process(
                target=single_forward,
                args=(eigenvalues[i], eigenvectors[i], dx[i],
                      star0.detach().cpu().numpy(),
                      star1.detach().cpu().numpy(),
                      d, n_eig, n_extra_eig, ctx.device))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        ctx.eigenvalues = eigenvalues
        ctx.eigenvectors = eigenvectors
        ctx.dx = dx

        ret = []
        for i in range(nb):
            nvec = inputs[3*i+1].shape[1]
            ret.extend([
                ctx.eigenvalues[i][nvec:n_eig],
                ctx.eigenvectors[i][:, nvec:n_eig]])

        return tuple(ret)

    @staticmethod
    def backward(ctx, *grad_output):
        ret = [None, None, None]

        for i in range(ctx.nb):
            # derivative wrt eigenvalues
            dstar0, dstar1 = single_backward(
                ctx.dx[i], ctx.eigenvalues[i], ctx.eigenvectors[i], ctx.n_eig,
                grad_output[2*i], grad_output[2*i+1], ctx.device)
            ret.extend([dstar0, dstar1, None])

        return tuple(ret)
