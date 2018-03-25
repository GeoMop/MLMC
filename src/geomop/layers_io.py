import sys
import os
import inspect
import json
import enum


import gm_base.json_data as js

"""
TODO:
1. every format is copy of the previous, so we have to convert ALL classes to the new ones even if they are same.
2. In order to use standard ser/deser. We can convert as:
   Region(region.serialize()), this can be implemented as a default convert method (using also recursion) in JSONData.
3. Keep conversions with individual formats. This means that format files should not be changed once the format is fixed.
4. We make conversion to the actual version, which can contain __eq__ methods etc. Futher changes in le_serializer 
   get rid of geometry_factory and auxiliary method in geometry_structures. In fact geometry_structures should be
5. Final state:
   - geometry structures keep latest state of the format common to LayerEditor and Geometry, without auxiliary methods
   - le_serializer have main and aux methods to convert between this format and internal structures, 
     data layer should be kept as close to the format structures as possible
   - Geometry creates objects with same structure, but with further deserialize steps which are now in "init" methdos. 
   This should be changed to copy data from format by singel common method, but do not derive classes from the 
   format classes. Make init functions to do the conversion, taking both format class and root obj as parameters. 
"""


def convert_to_0_4_9(layers, fmt):
    def region_fix_dim(region, extruded):
        if region.topo_dim != fmt.TopologyDim.invalid:
            # old format
            if region.dim == fmt.RegionDim.invalid:
                region.dim = fmt.RegionDim(region.topo_dim + extruded)
            if region.not_used:
                return
            assert region.dim.value == region.topo_dim + extruded, "Region {} , dimension mismatch."
        assert region.dim != fmt.RegionDim.invalid


    pass


def convert_to_0_5_0(layers, gs_new):
    pass

versions=[
    ([0,4,0], "format_0_4_0"),
    ([0,4,9], "format_0_4_9"),
    ([0,5,0], "format_0_5_0"),
    ([9,9,9], "format_last")
]





def module_only_import(name):
    """
    Import a module with version definition.
    Add default 'convert' method to all JsonData classes
    whithout explicit convert method.
    :param name:
    :return:
    """
    module = __import__(name, fromlist=['foo'])  # Just any name in 'fromlist' to trick the __import__function.
    for ele_name in dir(module):
        element = getattr(module, ele_name)
        if inspect.isclass(element) and ele_name != 'JsonData' and issubclass(element, js.JsonData):
            if not hasattr(element, 'convert'):

                @classmethod
                def dflt_conv(cls, x):
                    return convert_json_data(module, x)
                #print("set %s"%(str(element)))
                setattr(element, 'convert', dflt_conv)
    return module


def convert_json_data(module, old_version_object, class_obj=None):
    old_obj = old_version_object
    if old_obj is None:
        return None

    if class_obj is None:
        class_name = old_obj.__class__.__name__
        class_obj = getattr(module, class_name)

    new_obj = class_obj.__new__(class_obj)
    for key, item in old_obj.__dict__.items():
        item = convert_object(module, item)
        new_obj.__dict__[key] = item
    return new_obj


def convert_object(module, old_version_object):
    """
    Default convert method can be used only if the class in the old version is converted by taking
    its counter part in the new version and can copy data recursively, where all child classes must
    exist in the new version as well.
    :param module:
    :param old_version_object:
    :return:
    """
    old_obj = old_version_object
    if issubclass(old_obj.__class__, js.JsonData):
        class_name = old_obj.__class__.__name__
        class_obj = getattr(module, class_name)
        assert hasattr(class_obj, "convert")
        convert_fn = getattr(class_obj, "convert")
        new_obj = convert_fn(old_obj)
    elif isinstance(old_obj, (list, tuple)):
        new_item = [ convert_object(module, i) for i in old_obj ]
        new_obj = old_obj.__class__(new_item)
    elif isinstance(old_obj, dict):
        for k, v in old_obj.items():
            old_obj[k] = convert_object(module, v)
        new_obj = old_obj
    else:
        obj_class = old_obj.__class__
        if issubclass(obj_class, enum.IntEnum):
            class_obj = getattr(module, obj_class.__name__)
            new_obj = class_obj(old_obj)
        else:
            new_obj = old_obj
    return new_obj


def convert_file_to_actual_format(json_obj, base_path=""):
    version = json_obj.get("version", [0, 4, 0])
    layers = None
    for ver, format_module in versions:
        if version == ver:
            gs = module_only_import("gm_base.geometry_files." + format_module)
            layers = gs.LayerGeometry(json_obj)
        if version < ver:
            if layers is None:
                raise Exception("Unknown version of the layers file: %s"%(str(version)))
            gs_new_module = module_only_import("gm_base.geometry_files." + format_module)
            layers.base_path = base_path
            layers = gs_new_module.LayerGeometry.convert(layers)

    del layers.base_path
    return layers


def read_geometry(file_name):
    """return LayerGeometry data"""
    with open(file_name) as f:
        contents = f.read()
    json_lg = json.loads(contents, encoding="utf-8")
    base_path = os.path.dirname(file_name)
    return convert_file_to_actual_format(json_lg, base_path=base_path)
    return lg


def write_geometry(file_name, lg):
    """Write LayerGeometry data to file"""
    with open(file_name, 'w') as f:
        json.dump(lg.serialize(), f, indent=4, sort_keys=True)

