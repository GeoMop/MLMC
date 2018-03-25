"""Structures for Layer Geometry File"""

import sys
import os



from gm_base.json_data import *


class LayerType(IntEnum):
    """Layer type"""
    stratum = 0
    fracture = 1
    shadow = 2
    

class TopologyType(IntEnum):
    given = 0
    interpolated = 1


class RegionDim(IntEnum):
    invalid = -2
    none = -1
    point = 0
    well = 1
    fracture = 2
    bulk = 3
    

class TopologyDim(IntEnum):
    invalid = -1
    node = 0
    segment = 1
    polygon = 2


class Curve(JsonData):
    def __init__(self, config={}):
        super().__init__(config)


class SurfaceApproximation(JsonData):
    """
    Serialization class for Z_Surface.
    """
    def __init__(self, config={}):
        self.u_knots = [float]
        self.v_knots = [float]
        self.u_degree = 2
        self.v_degree = 2
        self.rational = False
        self.poles = [ [ [float] ] ]
        self.orig_quad = 4*(2*(float,),)
        self.xy_map = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        self.z_map = [1.0, 0.0]
        super().__init__(config)


class Surface(JsonData):
    
    def __init__(self, config={}):
        self.grid_file = ""
        """File with approximated points (grid of 3D points). None for plane"""
        self.name = ""
        """Surface name"""
        self.approximation = ClassFactory(SurfaceApproximation)
        """Serialization of the  Z_Surface."""
        super().__init__(config)
        
    # @staticmethod
    # def make_surface():
    #     surf = Surface()
    #     surf.approximation = None
    #     return surf

class Interface(JsonData):
    
    def __init__(self, config={}):
        self.surface_id = int
        """Surface index"""
        self.transform_z = 2*(float,)
        """Transformation in Z direction (scale and shift)."""
        self.elevation = float
        """ Representative Z coord of the surface."""

        # Grid polygon should be in SurfaceApproximation, however
        # what for the case of planar interfaces without surface reference.
        #self.grid_polygon = 4*(2*(float,))
        """Vertices of the boundary polygon of the grid."""
        super().__init__(config)


    def __eq__(self, other):
        """operators for comparation"""
        return self.elevation == other.elevation \
            and self.transform_z == other.transform_z \
            and self.surface_id != other.surface_id



class Segment(JsonData):

    """Line object"""
    def __init__(self, config={}):
        self.node_ids  = ( int, int )
        """First point index"""
        """Second point index"""
        self.interface_id = None
        """Interface index"""
        super().__init__(config)

    def __eq__(self, other):
        return self.node_ids == other.node.ids \
            and self.surface_id == other.surface_id

class Polygon(JsonData):

    """Polygon object"""
    def __init__(self, config={}):
        self.segment_ids = [ int ]
        """List of segments index of the outer wire."""
        self.holes = []
        """List of lists of segments of hole's wires"""
        self.free_points = [ int ]
        """List of free points in polygon."""
        self.interface_id = None
        """Interface index"""
        super().__init__(config)

    def __eq__(self, other):
        return self.segment_ids == other.segment_ids \
            and self.holes == other.holes \
            and self.free_points == other.free_points \
            and self.surface_id == other.surface_id


class Topology(JsonData):
    """Topological presentation of geometry objects"""

    def __init__(self, config={}):
        self.segments = [ ClassFactory(Segment) ]
        """List of topology segments (line)"""
        self.polygons = [ ClassFactory(Polygon) ]
        """List of topology polygons"""
        super().__init__(config)

    def __eq__(self, other):
        return self.segments == other.segments \
            and self.polygons == other.polygons \



class NodeSet(JsonData):

    """Set of point (nodes) with topology"""

    def __init__(self, config={}):
        self.topology_id = int
        """Topology index"""
        self.nodes = [ (float, float) ]
        """list of Nodes"""
        self.linked_node_set_id = None
        """node_set_idx of pair interface node set or None"""
        self.linked_node_ids = [ ]
        """List of node IDs that match node ids in other nodesets on the same interface. I.e. arbitrary number of nodesets can be linkedIf linked_node_set is not None there is list od pair indexes of nodes or none
        if node has not pair"""
        super().__init__(config)

    def reset(self):
        """Reset node set"""
        self.nodes = []


class InterfaceNodeSet(JsonData):
    """Node set in space for transformation(x,y) ->(u,v). 
    Only for GL"""
    _not_serialized_attrs_ = ['interface_type']

    def __init__(self, config={}):
        self.nodeset_id = int
        """Node set index"""
        self.interface_id = int
        """Interface index"""
        super().__init__(config)
        self.interface_type = TopologyType.given


class InterpolatedNodeSet(JsonData):
    """Two node set with same Topology in space for transformation(x,y) ->(u,v).
    If both node sets is same, topology is vertical    
    Only for GL"""
    _not_serialized_attrs_ = ['interface_type']

    def __init__(self, config={}):
        self.surf_nodesets = ( ClassFactory([InterfaceNodeSet]), ClassFactory([InterfaceNodeSet]) )
        """Top and bottom node set index"""
        self.interface_id = int
        """Interface index"""
        super().__init__(config)
        self.interface_type = TopologyType.interpolated


class Region(JsonData):
    """Description of disjunct geometri area sorte by dimension (dim=1 well, dim=2 fracture, dim=3 bulk). """
    
    def __init__(self, config={}):
        self.color = ""
        """8-bite region color"""
        self.name = ""
        """region name"""
        self.dim = RegionDim.invalid
        """ Real dimension of the region. (0,1,2,3)"""
        self.topo_dim = TopologyDim.invalid
        """For backward compatibility. Dimension (0,1,2) in Stratum layer: node, segment, polygon"""
        self.boundary = False
        """Is boundary region"""
        self.not_used = False
        """is used """
        self.mesh_step = 0.0
        """mesh step - 0.0 is automatic choice"""
        self.brep_shape_ids = [ ]
        """List of shape indexes - in BREP geometry """
        super().__init__(config)

    def fix_dim(self, extruded):

        if self.topo_dim != TopologyDim.invalid:
            # old format
            if self.dim == RegionDim.invalid:
                self.dim = RegionDim(self.topo_dim + extruded)
            if self.not_used:
                return
            assert self.dim.value == self.topo_dim + extruded, "Region {} , dimension mismatch."
        assert self.dim != RegionDim.invalid


class GeoLayer(JsonData):
    """Geological layers"""
    _not_serialized_attrs_ = ['layer_type']

    def __init__(self, config={}):
        self.name =  ""
        """Layer Name"""

        self.top =  ClassFactory( [InterfaceNodeSet, InterpolatedNodeSet] )
        """Accoding topology type interface node set or interpolated node set"""
        
        # assign regions to every topology object
        self.polygon_region_ids = [ int ]
        self.segment_region_ids = [ int ]
        self.node_region_ids = [ int ]

        super().__init__(config)
        self.layer_type = LayerType.shadow

    def fix_region_dim(self, regions):
        extruded = (self.layer_type == LayerType.stratum)
        for reg_list in  [self.polygon_region_ids, self.segment_region_ids, self.node_region_ids]:
            for reg_idx in reg_list:
                if reg_idx>0:
                    reg = regions[reg_idx]
                    reg.fix_dim(extruded)
                
    def fix_region_id(self):
        for reg_list in  [self.polygon_region_ids, self.segment_region_ids, self.node_region_ids]:
            for i in range(0, len(reg_list)):
                if reg_list[i]>2:
                    reg_list[i] -= 2
                else:
                    reg_list[i] = 0


class FractureLayer(GeoLayer):
    _not_serialized_attrs_ = ['layer_type', 'top_type']

    def __init__(self, config={}):
        super().__init__(config)
        self.layer_type = LayerType.fracture
        self.top_type = self.top.interface_type
class StratumLayer(GeoLayer):
    _not_serialized_attrs_ = ['layer_type', 'top_type','bottom_type']

    def __init__(self, config={}):

        self.bottom = ClassFactory( [InterfaceNodeSet, InterpolatedNodeSet] )
        """ optional, only for stratum type, accoding bottom topology
        type interface node set or interpolated node set"""

        super().__init__(config)
        self.layer_type = LayerType.stratum
        self.top_type = self.top.interface_type
        self.bottom_type = self.bottom.interface_type


class ShadowLayer(GeoLayer):
    def __init__(self, config={}):
        super().__init__(config)


class UserSupplement(JsonData):
    def __init__(self, config={}):
        self.last_node_set = 0
        """Last edited node set"""
        self.init_area = [(0.0, 0.0),  (1.0, 0.0),  (1.0, 1.0),  (0.0, 1.0)]
        """Initialization area (polygon x,y coordinates)"""
        self.zoom = {'zoom':1.0, 'x':0.0, 'y':0.0, 'position_set':False}  
        """Zoom and position for zoom diagram class""" 
        self.shps = [] 
        """Zoom and position for zoom diagram class"""
        self.surface_idx = None
        """Surface idx displayed surface panel""" 
        super().__init__(config)


class LayerGeometry(JsonData):
    def __init__(self, config={}):
        self.version = [0,4,0]
        """Version of the file format."""
        self.regions = [ ClassFactory(Region) ]
        """List of regions"""
        self.layers = [ ClassFactory( [StratumLayer, FractureLayer] ) ]
        """List of geological layers"""
        self.surfaces = [ ClassFactory(Surface) ]
        """List of B-spline surfaces"""
        self.interfaces = [ ClassFactory(Interface) ]
        """List of interfaces"""
        self.curves = [ ClassFactory(Curve) ]
        """List of B-spline curves,"""
        self.topologies = [ ClassFactory(Topology) ]
        """List of topologies"""
        self.node_sets = [ ClassFactory( NodeSet) ]
        """List of node sets"""
        self.supplement = UserSupplement()
        """Addition data that is used for displaying in layer editor"""
        super().__init__(config)

