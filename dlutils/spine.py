
import numpy as np
from neuron import h

__all__ = ['Spine']


def point_at_distance(origin, dst, m, q):
    v1 = origin
    v2 = np.array([v1[0]+1, m*(v1[0]+1) + q])
    u = (v2 - v1) / np.linalg.norm(v2 - v1)
    v3 = v1 + dst * u
    return v3


def make_spine_coords(points, lengths):
    n_points = int((points.shape[0] - 1) / 2)
    n_dims = points.shape[1]
    center = points[n_points,:]
    if np.all(points[:,0] == points[0,0]):
        m = 0
    elif np.all(points[:,1] == points[0,1]):
        raise NotImplementedError('Dendrite cannot be parallel to the x axis')
    else:
        p = np.polyfit(points[:,0], points[:,1], 1)
        m = -1 / p[0]
    q = center[1] - m * center[0]
    n_points = len(lengths)
    spine_points = np.zeros((n_points, n_dims))
    for i in range(n_points):
        spine_points[i,:2] = point_at_distance(center[:2], lengths[i], m, q)
    if n_dims == 3:
        spine_points[:,2] = center[2]
    return spine_points


class Spine (object):
    def __init__(self, sec, x, head_L, head_diam, neck_L, neck_diam=None, Ra=None, spine_id=None):
        n_points = sec.n3d()
        coords = np.zeros((n_points, 3))
        diams = np.zeros(n_points)
        norm_arclength = np.zeros(n_points)
        for i in range(n_points):
            coords[i,:] = np.array([sec.x3d(i),\
                                    sec.y3d(i),\
                                    sec.z3d(i)])
            diams[i] = sec.diam3d(i)
            norm_arclength[i] = sec.arc3d(i) / sec.L
        idx = np.argmin(np.abs(norm_arclength - x))
        N = 3
        start = np.max([idx-N, 0])
        stop = np.min([idx+N+1, coords.shape[0]-1])
        points = coords[start : stop, :]
        lengths = diams[idx] + np.array([0, neck_L, neck_L, head_L+neck_L])
        self._points = make_spine_coords(points, lengths)
        if neck_diam is None:
            neck_diam = diams[idx]
        self._diams = np.array([neck_diam, neck_diam, head_diam, head_diam])
        self._sec = sec
        self._sec_x = norm_arclength[idx]

        if Ra is not None:
            self._Ra = Ra
        else:
            self._Ra = self._sec.Ra

        if spine_id is not None:
            self._id = '-{}'.format(spine_id)
        else:
            self._id = ''

    def instantiate(self):
        self.neck = h.Section(name = 'neck' + self._id)
        self.head = h.Section(name = 'head' + self._id)
        self.neck.nseg = 1
        self.head.nseg = 1
        self.geometry()
        self.connect()
        self.biophysics()

    def geometry(self):
        # spine neck
        xvec = h.Vector(self._points[:2,0])
        yvec = h.Vector(self._points[:2,1])
        zvec = h.Vector(self._points[:2,2])
        dvec = h.Vector(self._diams[:2])
        h.pt3dadd(xvec, yvec, zvec, dvec, sec=self.neck)
        # spine head
        xvec = h.Vector(self._points[2:4,0])
        yvec = h.Vector(self._points[2:4,1])
        zvec = h.Vector(self._points[2:4,2])
        dvec = h.Vector(self._diams[2:4])
        h.pt3dadd(xvec, yvec, zvec, dvec, sec=self.head)

    def connect(self):
        self.head.connect(self.neck)
        self.neck.connect(self._sec(self._sec_x))

    def biophysics(self):
        print('Spine axial resistivity: {:.2f} Ohm cm.'.format(self._Ra))
        for sec in (self.neck, self.head):
            sec.cm = self._sec.cm
            sec.Ra = self._Ra
            sec.insert('pas')
            sec.g_pas = self._sec(self._sec_x).g_pas
            sec.e_pas = self._sec(self._sec_x).e_pas
