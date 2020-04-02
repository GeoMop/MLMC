"""
Dynmaicaly adding end removing AABB (axis aligned bounding boxes).
Supported operations:
- closest to point
- intersecting box

Implementation:
- linear search with usage of numpy arrays
Possible improvements using some trees.
"""
import numpy as np
#import numpy.linalg as la

_blow_box = np.array([-1.0, -1.0, 1.0, 1.0])
def make_aabb(points, margin=-1):
    """
    Make AABB of a set of points, optionaly expanded by given margin.
    :param points: Iterable of np array points [x,y]
    :param margin: width of added margin
    :return: box as np array [min_x, min_y, max_x, max_y]
    """
    points = np.array(points, dtype=float)
    box = np.concatenate( (np.min(points, axis=0), np.max(points, axis=0)) )
    if margin > 0:
        box += margin * _blow_box
    return box

class AABB_Lookup:
    def __init__(self, infty = 1e50, init_size=128):
        self.inf = infty
        self.n_boxes = 0
        self.boxes = np.full((init_size, 4), self.inf)

    def add_object(self, id, box):
        """
        Add a new object as set of boxes. Any original box with same ID is replaced.
        :param id: Object ID, we assume nearly continuous IDs.
        :param box: np array [min_x, min_y, max_x, max_y]
        :return: None
        """
        boxes_size = self.boxes.shape[0]
        while id >= boxes_size:
            # double the size
            self.boxes = np.append( self.boxes, np.full((boxes_size, 4), self.inf), axis=0 )
            boxes_size = self.boxes.shape[0]
        self.n_boxes = max(self.n_boxes, id + 1)
        self.boxes[id, :] = box

    def rm_object(self, id):
        self.boxes[id, :] = self.inf

    def closest_candidates(self, point):
        """
        Return IDs of boxes that may contain boxes closest to the given point
        in L2 norm.
        :param point: np array [x,y]
        :return: List of IDs.
        """
        boxes = self.boxes[:self.n_boxes, :]
        if boxes.shape[0] == 0 :
            return []
        inf_dists = np.max(np.maximum(self.boxes[:, 0:2] - point, point - self.boxes[:, 2:4]), axis=1)
        if np.amin(inf_dists) > 0.0:
            # closest box not containing the point
            i_closest = np.argmin(inf_dists)
            c_boxes = boxes[i_closest:i_closest+1, :]
        else:
            # point is inside all boxes
            c_boxes = boxes[np.where(inf_dists<=0.0)]
        assert c_boxes.shape[0] != 0
        # Max distance of closest boxes
        #try:
        l_inf_max = np.max(np.maximum(point - c_boxes[:, 0:2], c_boxes[:, 2:4] - point))
        #except:
        #    pass
        l2_max = np.sqrt(2) * l_inf_max
        return np.where(inf_dists < l2_max)[0]


    def intersect_candidates(self, box):
        """
        :param box: np array [min_x, min_y, max_x, max_y]
        :return: List of ids of boxes that intersect with given box.
        """
        boxes = self.boxes[:self.n_boxes, :]
        not_intersect = np.logical_or(
                            box[2: 4] < boxes[:, 0:2],
                            boxes[:, 2:4] < box[0:2])
        not_intersect = np.logical_or(not_intersect[:,0], not_intersect[:,1])
        return np.where( np.logical_not(not_intersect) )[0]