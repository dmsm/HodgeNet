import igl
import numpy as np
import scipy
import scipy.io
import torch
import trimesh


def random_rotation_matrix():
    """Generate a random 3D rotation matrix."""
    Q, _ = np.linalg.qr(np.random.normal(size=(3, 3)))
    return Q


def random_scale_matrix(max_stretch):
    """Generate a random 3D anisotropic scaling matrix."""
    return np.diag(1 + (np.random.rand(3)*2 - 1) * max_stretch)


def d01(v, e):
    """Compute d01 operator from 0-froms to 1-forms."""
    row = np.tile(np.arange(e.shape[0]), 2)
    col = e.T.flatten()
    data = np.concatenate([np.ones(e.shape[0]), -np.ones(e.shape[0])], axis=0)
    d = scipy.sparse.csr_matrix(
        (data, (row, col)), dtype=np.double, shape=(e.shape[0], v.shape[0]))
    return d


def flip(f, ev, ef, bdry=False):
    """Compute vertices of flipped edges."""
    # glue together the triangle vertices adjacent to each edge
    duplicate = f[ef].reshape(ef.shape[0], -1)
    duplicate[(duplicate == ev[:, 0, None])
              | (duplicate == ev[:, 1, None])] = -1  # remove edge vertices

    # find the two remaining verts (not -1) in an orientation-preserving way
    idxs = (-duplicate).argsort(1)[:, :(1 if bdry else 2)]
    idxs.sort(1)  # preserve orientation by doing them in index order
    result = np.take_along_axis(duplicate, idxs, axis=1)

    return result


class HodgenetMeshDataset(torch.utils.data.Dataset):
    """Dataset of meshes with labels."""

    def __init__(self, mesh_files, mesh_features={}, decimate_range=None,
                 random_rotation=True, max_stretch=0.1,
                 edge_features_from_vertex_features=['vertices'],
                 triangle_features_from_vertex_features=['vertices'],
                 center_vertices=True, normalize_coords=True,
                 segmentation_files=None):
        self.mesh_files = mesh_files
        self.mesh_features = mesh_features
        self.decimate_range = decimate_range
        self.random_rotation = random_rotation
        self.max_stretch = max_stretch
        self.edge_features_from_vertex_features = \
            edge_features_from_vertex_features
        self.triangle_features_from_vertex_features = \
            triangle_features_from_vertex_features
        self.center_vertices = center_vertices
        self.segmentation_files = segmentation_files
        self.normalize_coords = normalize_coords

        self.min_category = float('inf')
        self.n_seg_categories = 0
        if self.segmentation_files is not None:
            for f in segmentation_files:
                triangle_data = np.loadtxt(f, dtype=np.int64)
                self.n_seg_categories = max(
                    self.n_seg_categories, triangle_data.max())
                self.min_category = min(self.min_category, triangle_data.min())

            self.n_seg_categories = self.n_seg_categories-self.min_category+1

    def __len__(self):
        return len(self.mesh_files)

    def __getitem__(self, idx):
        mesh = trimesh.load(self.mesh_files[idx], process=False)
        v_orig, f_orig = mesh.vertices, mesh.faces

        if self.segmentation_files is not None:
            face_segmentation = np.loadtxt(self.segmentation_files[idx],
                                           dtype=int)
            face_segmentation -= self.min_category

        # decimate mesh to desired number of faces in provided range
        if self.decimate_range is not None:
            while True:
                nfaces = np.random.randint(
                    min(f_orig.shape[0], self.decimate_range[0]),
                    min(f_orig.shape[0], self.decimate_range[1]) + 1)

                _, v, f, decimated_f_idxs, _ = igl.decimate(
                    v_orig, f_orig, nfaces)
                if igl.is_edge_manifold(f):
                    break

            if self.segmentation_files is not None:
                face_segmentation = face_segmentation[decimated_f_idxs]
        else:
            v = v_orig
            f = f_orig

        if self.center_vertices:
            v -= v.mean(0)

        if self.normalize_coords:
            v /= np.linalg.norm(v, axis=1).max()

        # random rotation/scaling
        if self.random_rotation:
            v = v @ random_rotation_matrix()
        if self.max_stretch != 0:
            v = v @ random_scale_matrix(self.max_stretch)
        if self.random_rotation:
            v = v @ random_rotation_matrix()

        areas = igl.doublearea(v, f)

        ev, fe, ef = igl.edge_topology(v, f)

        bdry_idxs = (ef == -1).any(1)
        if bdry_idxs.sum() > 0:
            bdry_ev = ev[bdry_idxs]
            bdry_ef = ef[bdry_idxs]
            bdry_ef.sort(1)
            bdry_ef = bdry_ef[:, 1, None]
            bdry_flipped = flip(f, bdry_ev, bdry_ef, bdry=True)

        int_ev = ev[~bdry_idxs]
        int_ef = ef[~bdry_idxs]
        int_flipped = flip(f, int_ev, int_ef)

        # normals
        n = igl.per_vertex_normals(v, f)
        n[np.isnan(n)] = 0

        result = {
            'vertices': torch.from_numpy(v),
            'faces': torch.from_numpy(f),
            'areas': torch.from_numpy(areas),
            'int_d01': d01(v, int_ev),
            'triangles': torch.from_numpy(f.astype(np.int64)),
            'normals': torch.from_numpy(n),
            'mesh': self.mesh_files[idx],
            'int_ev': torch.from_numpy(int_ev.astype(np.int64)),
            'int_flipped': torch.from_numpy(int_flipped.astype(np.int64)),
            'f': torch.from_numpy(f.astype(np.int64))
        }

        if self.segmentation_files is not None:
            result['segmentation'] = torch.from_numpy(face_segmentation)
        if bdry_idxs.sum() > 0:
            result['bdry_d01'] = d01(v, bdry_ev)

        # feature per mesh (e.g. label of the mesh)
        for key in self.mesh_features:
            result[key] = self.mesh_features[key][idx]

        # gather vertex features to edges from list of keys
        result['int_edge_features'] = torch.from_numpy(
            np.concatenate([
                np.concatenate([
                    result[key][torch.from_numpy(int_ev.astype(np.int64))],
                    result[key][torch.from_numpy(int_flipped.astype(np.int64))]
                ], axis=1).reshape(int_ev.shape[0], -1)
                for key in self.edge_features_from_vertex_features
            ], axis=1))

        if bdry_idxs.sum() > 0:
            result['bdry_edge_features'] = torch.from_numpy(
                np.concatenate([
                    np.concatenate([
                        result[key][torch.from_numpy(
                            bdry_ev.astype(np.int64))],
                        result[key][torch.from_numpy(
                            bdry_flipped.astype(np.int64))]
                    ], axis=1).reshape(bdry_ev.shape[0], -1)
                    for key in self.edge_features_from_vertex_features
                ], axis=1))

        # gather vertex features to triangles from list of keys
        result['triangle_features'] = torch.from_numpy(
            np.concatenate(
                [result[key][f].reshape(f.shape[0], -1)
                 for key in self.triangle_features_from_vertex_features],
                axis=1))

        return result


def get_rot(theta):
    """Get 3D rotation matrix for a given angle."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])


class OrigamiDataset(torch.utils.data.Dataset):
    """Dataset of square mesh with random crease down the center."""

    def __init__(self, edge_features_from_vertex_features=['vertices'],
                 triangle_features_from_vertex_features=['vertices']):
        self.edge_features_from_vertex_features = \
            edge_features_from_vertex_features
        self.triangle_features_from_vertex_features = \
            triangle_features_from_vertex_features

        self.v, self.f = igl.read_triangle_mesh('square.obj')

    def __len__(self):
        return 5000

    def __getitem__(self, _):
        v, f = self.v, self.f

        v1 = v[:55]
        v2 = v[55:]

        theta = np.random.rand() * 2 * np.pi
        v2_ = v2 @ get_rot(np.pi - theta)
        v = np.concatenate([v1, v2_], axis=0)

        ev, _, ef = igl.edge_topology(v, f)

        bdry_idxs = (ef == -1).any(1)
        bdry_ev = ev[bdry_idxs]
        bdry_ef = ef[bdry_idxs]
        bdry_ef.sort(1)
        bdry_ef = bdry_ef[:, 1, None]

        bdry_flipped = flip(f, bdry_ev, bdry_ef, bdry=True)

        int_ev = ev[~bdry_idxs]
        int_ef = ef[~bdry_idxs]

        int_flipped = flip(f, int_ev, int_ef)

        # normals
        n = igl.per_vertex_normals(v, f)

        result = {
            'vertices': torch.from_numpy(v),
            'int_d01': d01(v, int_ev),
            'bdry_d01': d01(v, bdry_ev),
            'triangles': torch.from_numpy(f.astype(np.int64)),
            'normals': torch.from_numpy(n),
            'dir': torch.tensor([np.cos(theta), np.sin(theta)]),
        }

        # gather vertex features to edges from list of keys
        result['int_edge_features'] = torch.from_numpy(
            np.concatenate([
                np.concatenate([
                    result[key][torch.from_numpy(int_ev)],
                    result[key][torch.from_numpy(int_flipped)]
                ], axis=1).reshape(int_ev.shape[0], -1)
                for key in self.edge_features_from_vertex_features
            ], axis=1))
        result['bdry_edge_features'] = torch.from_numpy(
            np.concatenate([
                np.concatenate([
                    result[key][torch.from_numpy(bdry_ev)],
                    result[key][torch.from_numpy(bdry_flipped)]
                ], axis=1).reshape(bdry_ev.shape[0], -1)
                for key in self.edge_features_from_vertex_features
            ], axis=1))

        # gather vertex features to triangles from list of keys
        result['triangle_features'] = torch.from_numpy(np.concatenate(
            [result[key][f].reshape(f.shape[0], -1)
             for key in self.triangle_features_from_vertex_features], axis=1))

        return result
