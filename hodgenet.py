import scipy
import scipy.sparse.linalg
import torch
import torch.nn as nn

from hodgeautograd import HodgeEigensystem


class HodgeNetModel(nn.Module):
    """Main HodgeNet model.

    The model inputs a batch of meshes and outputs features per vertex or 
    pooled to faces or the entire mesh.
    """
    def __init__(self, num_edge_features, num_triangle_features,
                 num_output_features=32, num_eigenvectors=64,
                 num_extra_eigenvectors=16, mesh_feature=False, min_star=1e-2,
                 resample_to_triangles=False, num_bdry_edge_features=None,
                 num_vector_dimensions=1):
        super(HodgeNetModel, self).__init__()

        self.num_triangle_features = num_triangle_features
        self.hodgefunc = HodgeEigensystem.apply
        self.num_eigenvectors = num_eigenvectors
        self.num_extra_eigenvectors = num_extra_eigenvectors
        self.num_output_features = num_output_features
        self.min_star = min_star
        self.resample_to_triangles = resample_to_triangles
        self.mesh_feature = mesh_feature
        self.num_vector_dimensions = num_vector_dimensions

        self.to_star1 = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, self.num_vector_dimensions**2)
        )

        if num_bdry_edge_features is not None:
            self.to_star1_bdry = nn.Sequential(
                nn.Linear(num_bdry_edge_features, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Linear(32, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Linear(32, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Linear(32, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Linear(32, self.num_vector_dimensions**2)
            )
        else:
            self.to_star1_bdry = None

        self.to_star0_tri = nn.Sequential(
            nn.Linear(num_triangle_features, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, self.num_vector_dimensions *
                      self.num_vector_dimensions)
        )

        self.eigenvalue_to_matrix = nn.Sequential(
            nn.Linear(1, num_output_features),
            nn.BatchNorm1d(num_output_features),
            nn.LeakyReLU(),
            nn.Linear(num_output_features, num_output_features),
            nn.BatchNorm1d(num_output_features),
            nn.LeakyReLU(),
            nn.Linear(num_output_features, num_output_features),
            nn.BatchNorm1d(num_output_features),
            nn.LeakyReLU(),
            nn.Linear(num_output_features, num_output_features),
            nn.BatchNorm1d(num_output_features),
            nn.LeakyReLU(),
            nn.Linear(num_output_features, num_output_features)
        )

    def gather_star0(self, mesh, star0_tri):
        """Compute star0 matrix per vertex by gathering from triangles."""
        star0 = torch.zeros(mesh['vertices'].shape[0],
                            star0_tri.shape[1]).to(star0_tri)
        star0.index_add_(0, mesh['triangles'][:, 0], star0_tri)
        star0.index_add_(0, mesh['triangles'][:, 1], star0_tri)
        star0.index_add_(0, mesh['triangles'][:, 2], star0_tri)

        star0 = star0.view(-1, self.num_vector_dimensions,
                           self.num_vector_dimensions)

        # square the tensor to be semidefinite
        star0 = torch.einsum('ijk,ilk->ijl', star0, star0)

        # add min star down the diagonal
        star0 += torch.eye(self.num_vector_dimensions)[None].to(star0) * \
            self.min_star

        return star0

    def compute_mesh_eigenfunctions(self, mesh, star0, star1, bdry=False):
        """Compute eigenvectors and eigenvalues of the learned operator."""
        nb = len(mesh)

        inputs = []
        for m, s0, s1 in zip(mesh, star0, star1):
            d = m['int_d01']
            if bdry:
                d = scipy.sparse.vstack([d, m['bdry_d01']])
            inputs.extend([s0, s1, d])

        eigenvalues, eigenvectors = [], []
        outputs = self.hodgefunc(nb, self.num_eigenvectors,
                                 self.num_extra_eigenvectors, *inputs)
        for i in range(nb):
            eigenvalues.append(outputs[2*i])
            eigenvectors.append(outputs[2*i+1])

        return eigenvalues, eigenvectors

    def forward(self, batch):
        nb = len(batch)

        all_star0_tri = self.to_star0_tri(
            torch.cat([mesh['triangle_features'] for mesh in batch], dim=0))
        star0_tri_split = torch.split(
            all_star0_tri, [mesh['triangles'].shape[0] for mesh in batch],
            dim=0)
        star0_split = [self.gather_star0(mesh, star0_tri)
                       for mesh, star0_tri in zip(batch, star0_tri_split)]

        all_star1 = self.to_star1(torch.cat([mesh['int_edge_features']
                                             for mesh in batch], dim=0))
        all_star1 = all_star1.view(-1, self.num_vector_dimensions,
                                   self.num_vector_dimensions)
        all_star1 = torch.einsum('ijk,ilk->ijl', all_star1, all_star1)
        all_star1 += torch.eye(
            self.num_vector_dimensions)[None].to(all_star1) * \
            self.min_star
        star1_split = list(torch.split(all_star1, [mesh['int_d01'].shape[0]
                                                   for mesh in batch], dim=0))

        if self.to_star1_bdry is not None:
            all_star1_bdry = self.to_star1_bdry(
                torch.cat([mesh['bdry_edge_features'] for mesh in batch],
                          dim=0))
            all_star1_bdry = all_star1_bdry.view(
                -1, self.num_vector_dimensions, self.num_vector_dimensions)
            all_star1_bdry = torch.einsum(
                'ijk,ilk->ijl', all_star1_bdry, all_star1_bdry)
            all_star1_bdry += torch.eye(
                self.num_vector_dimensions)[None].to(all_star1_bdry) * \
                self.min_star
            star1_bdry_split = torch.split(
                all_star1_bdry,
                [mesh['bdry_d01'].shape[0] for mesh in batch], dim=0)

            for i in range(nb):
                star1_split[i] = torch.cat(
                    [star1_split[i], star1_bdry_split[i]], dim=0)

        eigenvalues, eigenvectors = self.compute_mesh_eigenfunctions(
            batch, star0_split, star1_split,
            bdry=self.to_star1_bdry is not None)

        # glue the eigenvalues back together and run through the nonlinearity
        all_processed_eigenvalues = self.eigenvalue_to_matrix(
            torch.stack(eigenvalues).view(-1, 1)).view(
                nb, -1, self.num_output_features)

        # post-multiply the set of eigenvectors by the learned matrix that's a
        # function of eigenvalues (similar to HKS, WKS)
        outer_products = [torch.einsum(
            'ijk,ijl->ijkl', eigenvectors[i], eigenvectors[i])
            for i in range(nb)]  # take outer product of vectors

        result = [torch.einsum(
            'ijkp,jl->ilkp', outer_products[i], all_processed_eigenvalues[i])
            for i in range(nb)]  # multiply by learned matrix

        result = [result[i].flatten(start_dim=1) for i in range(nb)]

        if self.resample_to_triangles:
            result = [result[i][batch[i]['triangles']].max(
                1)[0] for i in range(nb)]

        if self.mesh_feature:
            result = [f.max(0, keepdim=True)[0] for f in result]

        return torch.cat(result, dim=0)
