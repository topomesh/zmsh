from math import comb as binomial
import numpy as np
import predicates
from .topology import Topology
from .geometry import Geometry
from . import simplicial
from .convex_hull import extreme_points as _extreme_points, ConvexHullMachine


def extreme_points(points: np.ndarray):
    r"""Given a `d`-dimensional point set, return the indices of `d + 1` points
    that are on the convex hull of the parabolic lift"""
    square_magnitudes = np.sum(points**2, axis=1)
    lifted_points = np.column_stack((points, square_magnitudes))
    return _extreme_points(lifted_points)


class DelaunayMachine:
    def __init__(self, points: np.ndarray):
        n = len(points)
        indices = extreme_points(points)

        d = points.shape[1]
        num_cells = [n] + [binomial(d + 1, k + 1) for k in range(1, d)] + [2]
        topology = Topology(dimension=d, num_cells=num_cells)
        geometry = Geometry(topology, points.copy())

        matrices = simplicial.simplex_to_chain_complex(d)[1:]
        cell_ids = [indices] + [tuple(range(D.shape[1])) for D in matrices]
        for k, D in enumerate(matrices, start=1):
            cells = geometry.topology.cells(k)
            cells[cell_ids[k - 1], cell_ids[k]] = D

        cells = geometry.topology.cells(d)
        cells[cell_ids[-2], 1] = -matrices[-1]

        self._convex_hull_machine = ConvexHullMachine(
            geometry, signed_volume=predicates.insphere
        )

    @property
    def geometry(self):
        return self._convex_hull_machine.geometry

    def is_done(self):
        return self._convex_hull_machine.is_done()

    def step(self):
        self._convex_hull_machine.step()

    def finalize(self):
        geometry = self._convex_hull_machine.finalize()
        self._convex_hull_machine = None

        topology = geometry.topology
        cells = topology.cells(topology.dimension)
        cell_ids_to_remove = []
        for cell_id in range(len(cells)):
            faces_ids, matrices = cells.closure(cell_id)
            orientation = simplicial.orientation(matrices)
            X = geometry.points[faces_ids[0], :]
            volume = orientation * predicates.volume(X.T)
            if volume > 0:
                cell_ids_to_remove.append(cell_id)

        D = topology.boundary(topology.dimension)
        for cell_id in cell_ids_to_remove:
            D[:, cell_id] = 0

        for k in range(topology.dimension - 1, 0, -1):
            cocells = topology.cocells(k)
            face_ids_to_remove = []
            for face_id, (cell_ids, signs) in enumerate(cocells):
                if len(cell_ids) == 0:
                    face_ids_to_remove.append(face_id)

            D = topology.boundary(k)
            for face_id in face_ids_to_remove:
                D[:, face_id] = 0

        for k in range(1, topology.dimension + 1):
            topology.cells(k).remove_empty_cells()

        return geometry

    def run(self):
        while not self.is_done():
            self.step()

        return self.finalize()


def delaunay(points):
    r"""Compute the Delaunay mesh of a point set"""
    machine = DelaunayMachine(points)
    return machine.run()


def _faces_cross(x, y):
    # TODO: Make this work in Rᵈ
    A = np.column_stack((x[:, 1] - x[:, 0], -(y[:, 1] - y[:, 0])))
    f = y[:, 0] - x[:, 0]
    s = np.linalg.solve(A, f)
    return (0 <= s).all() and (s <= 1).all()


class ConstrainedDelaunayMachine:
    def __init__(self, geometry: Geometry):
        if geometry.dimension != geometry.topology.dimension + 1:
            raise ValueError("Starting geometry must be a d-surface in Rᵈ⁺¹")

        if (geometry.topology.dimension != 1) or (geometry.dimension != 2):
            raise NotImplementedError(
                "Constrained Delaunay triangulation only implemented for 2D."
            )

        self._input_geometry = geometry
        self._geometry = delaunay(geometry.points)
        self._face_queue = list(range(len(geometry.topology.cells(1))))

    @property
    def face_queue(self):
        r"""The queue of constrained faces that still need to be added to the
        output geometry"""
        return self._face_queue

    def is_done(self):
        r"""Return `True` when there are no more constraining faces left to add
        and no more polygons to subdivide"""
        return not self._face_queue

    def crossing_faces(self, input_face_id):
        r"""Return the IDs of all faces in the current geometry that intersect
        a given face of the input geometry"""
        # Get the vertices in the input face
        dimension = self._input_geometry.dimension
        faces = self._input_geometry.topology.cells(dimension - 1)
        face_vertex_ids = faces.closure(input_face_id)[0][0]
        x2 = self._geometry.points[face_vertex_ids, :].T

        # Find all the faces and cells containing the input vertices
        covertices = self._geometry.topology.cocells(0)
        cocells_ids = covertices.closure(face_vertex_ids)[0]
        incident_face_ids = cocells_ids[-2]
        visited_cell_ids = cocells_ids[-1]

        # The starting faces are any faces of the initial cells not incident on
        # the input vertices
        faces = self._geometry.topology.cells(dimension - 1)
        cofaces = self._geometry.topology.cocells(dimension - 1)
        cells = self._geometry.topology.cells(dimension)
        face_ids = cells[visited_cell_ids][0]
        starting_face_ids = np.setdiff1d(face_ids, incident_face_ids)

        face_queue = set(starting_face_ids)
        result = np.array([], dtype=np.int64)
        while face_queue:
            face_id = face_queue.pop()
            vertex_ids = faces.closure(face_id)[0][0]
            x1 = self._geometry.points[vertex_ids, :].T
            if _faces_cross(x1, x2):
                result = np.append(result, [face_id])
                cell_ids = np.setdiff1d(cofaces[face_id][0], visited_cell_ids)
                neighbor_face_ids = cells[cell_ids][0]
                face_queue.update(np.setdiff1d(neighbor_face_ids, result))
                visited_cell_ids = np.append(visited_cell_ids, cell_ids)

        return result

    def step(self):
        raise NotImplementedError("Haven't got here yet!")

    def finalize(self):
        raise NotImplementedError("Haven't got here yet!")

    def run(self):
        while not self.is_done():
            self.step()

        return self.finalize()


def constrained_delaunay(geometry):
    return ConstrainedDelaunayMachine(geometry).run()
