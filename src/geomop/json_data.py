"""
json_data module provides base class for generic serialization/deserialiation
of derived classes with elementaty type checking.

Rules:
- classes are converted into dictionaries, tuples to the lists and scalar types (str, float, int, bool) and then
  serialized/deserialized  to/from JSON format.

- all attributes not starting with underscore are serialized by default. Class may provide list of attributes that should not be serialized
  or list of attributes that should be serialized to override this implicit mechanism.

- All attributes have to be created (in constructor) before serialization/deserialization. One can either assign default values
  in which case the input value must have the same type (checked recursively). Or just assign the type in which case the
  input value is checked against the type. If the value is not on input the value None is assigned.

- Attributes with type None can get any value from input, no checks or conversions are performed.

- None values propagates upwards through the recursion tree as follows:
    - [None ] -> []                         # Full sequence is [ int ] - no value on input -> [None] -> []
    - {a: None, b:x}  -> {a: None, b:x}     # nothing special
    - self.x = None  -> self..x = None      # nothing special

- Lists may have specified type of elements, setting just single element:
    [ int ]
    [ ClassFactory( [ A, B ] ) ]
    or default value with singel element (not very usefull)
    [ 0 ]

- Lists with no elements are treated as [ None ], i.e. list of anything.

- self.x = ClassFactory( class_list = [ A, B ] )
  Input for attribute x should be a dictionary, that should contain key '__class__' which
  must be name  of a class containd in the 'class_list'. The class of that name is constructed using deserialization recoursively.
  For class_lists of length 1, the '__class__' key is optional.

- self.x = Obligatory(<Type>)
  To mark an attribute with type <Type> that is obligatory on input.


Example:
import gm_base.json_data as jd

class Animal(jd.JsonData):
    def __init__(config = {}):
        self.n_legs = 4       # Default value, type given implicitely, optional on input.
        self.n_legs = int     # Just type, input value obligatory
        self.length = float   # floats are initializble also from ints

        self.head = Chicken   # Construct Chicken form the value
        self.head = jd.Factory({ Chicken, Duck, Goose}) # construct one of given types according to '__class__' key on input

        self.
        def_head = Chicken(dict( chicken_color: "brown") )
        self.head = jd.Factory([Chicken, Duck, Goose], default =def_head)


        super().__init__(config, fill_attrs = []) # run deserialization and checks
        # By default all public attributes (not starting with underscore) are
        # initialized. Alternatively the list of attributes may be provided by the fill_attrs parameter.



class Chicken(Animal):
    # explicit specification of attributes to serialize/deserialize.
    _serialized_attrs_ = [ 'color', 'tail' ]

    def __init__():
        def.color = 0
        def.wing = 1 # not serialized

TODO:

TODO:
- Track actual path in the data tree during recursive deserialization. Use this path in error messages.
- Distinguish asserts (check consistancy of code, can be safely removed in production code) and
input checks (these should be tested through explicit if (...): raise ...)
- Add unit tests for:
    - optional but typed input

FUTURE (as soon as we are not about to release):
- See existing solutions (attr - allow declaration of variables in the class and
  provides automatic generatrion of initialization, representation and other common methods.)

- Review design of the module so that definition of variables is part of the class, not part of the init,
  possibly using 'attr' in significant way.
    - automaticaly add methods to:
    - initialize to default values (should be called before explicit initialization)
    - deserialize (and validate) - specific kind of initialization
    - validate
    - serialize
    - repr == serialize ??

- Allow to serialize some objects into separate list converting references to indices. (support for the thing we do in Geometry)
-

- Instead of JsonDataNoConstruct have dict_type_of_class(my_class), which returns specification of
  dictionary representation of the class.

- Need to separate data validation and serialization/deserialization, default values. E.g.
  I need validation against [[int]], but default value []. But otherwise the default could be [[]], or None, ...


- Is __init__ the best place to specify types and do deserialization as well?
  We have no way to initialize private attributes (not accesible from input, but known at construction time of the class
  - object created dynamicaly at run time.
  Possibly we can convert this deserialization init into static factory method:

  @class
  def deserialize( cls, config ):
        x = cls.__new__()
        x.n_legs = 4
        # other type definitions
        super().deserialize(x, config )
        return x

  pros:
        - can add serialization/deserialization to already existing classes with its own constructors
        - allow explicti construction/initialization of the instancies
  cons:
        - Two places where we fill and possibly check class attributes

- Serialization of repeating objects:
        a = A()
        b = B(a)
        c = C(a)
        serialize ([b,c]) # whe should serialize 'a' just once and use kind of referencing (natural in YAML)




"""
#import json


#
#
# TODO:
#  - example of child classes
#  - support both types and default values in declaration of serializable attrs
#  - Just warn for unknown attrs on  input

from enum import IntEnum
import inspect






class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class WrongKeyError(Error):
    """Raised when attempt assign data to key that not exist"""
    def __init__(self, key):
        self.key = key

    def __str__(self):
        return "'{}'".format(self.key)


class Obligatory:
    """
    Wrapper to specify that the value is obligatory on input.
    """
    def __init__(self, type_spec):
        self.type=type_spec
    def type(self):
        return self.type


class ClassFactory:
    """
    Helper class for JsonData.
    """
    def __init__(self, class_list):
        """
        Initialize list of possible classes.
        :param class_list:
        """
        if type(class_list) != list:
            class_list = [ class_list ]

        self.class_list = class_list

    def make_instance(self, config, path):
        """
        Make instance from config dict.
        Dict must contain item "__class__" with name of desired class.
        Desired class must be in class_list.
        :param config:
        :return:
        """

        # Call default constructor when config is the default value
        if config.__class__ is not dict:
            if isinstance(config, ClassFactory):
                return None
            elif type(config) in self.class_list:
                return config
            else:
                raise Exception("Expecting dict instead of: \n{}\npath: {}".format(config, path))

        # __class__ specification not necessary for single valued 'class_list'
        if len(self.class_list) == 1 and  not "__class__" in config:
            config["__class__"] = self.class_list[0].__name__

        assert "__class__" in config, "Missing '__class__' key to construct one of: {}\npath: {}".format(self.config_list, path)
        class_name = config["__class__"]

        for c in self.class_list:
            if c.__name__ == class_name:
                config = config.copy()
                del config["__class__"]
                try:
                    return c(config)
                except TypeError:
                    raise TypeError("Non-standard JsonData constructor for class: {}\npath: {}".format(c, path))
                except:
                    raise Exception("Failed initialization of type: {}\npath: {}".format(c, path))
        assert False, "Input class: {} not in the factory list: {}\npath: {} ".format(class_name, self.class_list, path)

# class ClassFromList(ClassFactory):
#     def __init__(self, class_name):
#         assert issubclass(class_name, JsonData)
#         self.class_name = clase_name
#
#     def make_instance(self, config):
#         """
#         Make instance from config dict.
#         Dict must contain item "__class__" with name of desired class.
#         Desired class must be in class_list.
#         :param config:
#         :return:
#         """
#         assert config.__class__ is list
#         return class_name(config)


class JsonData:


    """
    Abstract base class for various data classes.
    These classes are basically just documented dictionaries,
    which are JSON serializable and provide some syntactic sugar
    (see DotDict from Flow123d - Jan Hybs)
    In order to simplify also serialization of non-data classes, we
    should implement serialization of __dict__.

    Why use JSON for serialization? (e.g. instead of pickle)
    We want to use it for both sending the data and storing them in files,
    while some of these files should be human readable/writable.

    Serializable classes will be derived from this one. And data members
    that should not be serialized are prefixed by '_'.

    If list of serialized attributes is provided to constructor,
    these attributes are serialized no matter if prefixed by '_' or not.

    ?? Anything similar in current JobPanel?
    """

    _serialized_attrs_ = []
    """ List of attributes to serialize. Leave empty to use public attributes."""

    _not_serialized_attrs_=[]
    """
    List of attributes to not serialize. Lower priarity then _serialized_attrs_. Leave empty to use non-public attributes.
    """

    def __init__(self, config):
        """
        Initialize class dict from config serialization.
        :param config: config dict
        :param serialized_attr: list of serialized attributes
        """

        if self._serialized_attrs_:
            self.__class__._not_serialized_attrs_ = [key for key in self.__dict__.keys() if key not in self._serialized_attrs_]
        elif self._not_serialized_attrs_:
            pass
        else:
            self.__class__._not_serialized_attrs_ = [ key  for key in self.__dict__.keys() if key[0] == "_" ]
        self.__class__._not_serialized_attrs_.extend( ['__class__', '_not_serialized_attrs_'] )

        path = []
        result_dict = self._deserialize_dict(self.__dict__, config, self._not_serialized_attrs_, path)
        for key, val in result_dict.items():
            self.__dict__[key] = val

    @staticmethod
    def _deserialize_dict(template_dict, config_dict, filter_attrs, path):
        result_dict = {}
        for key, temp in list(template_dict.items()):
            if key in filter_attrs:
                continue
            value = config_dict.get(key, temp)
            config_dict.pop(key, 0)

            assert not inspect.isclass(value), "Missing value for obligatory key '{}' of type: {}.".format(key, temp)
            filled_template = JsonData._deserialize_item(temp, value, path + [key])
            result_dict[key] = filled_template

        for key in ['__class__']:
            config_dict.pop(key, 0)

        if config_dict.keys():
            raise WrongKeyError("Keys {} not serializable attrs of dict:\n{}\n{}"
                                .format(list(config_dict.keys()), template_dict, path))
        return result_dict

    @staticmethod
    def _deserialize_item(temp, value, path):
        """
        Deserialize value.


        :param temp: template for assign value, just type for scalar types, dafualt value already assigned to value.
        :param value: value for deserialization
        :return:
        """
        if isinstance(temp, Obligatory):
            if value  is None:
                raise Exception("Missing obligatory key, path: {}".format(path))
            else:
                temp = temp.type


        # No check.
        if temp is None:
            return value

        # Explicitely no value for a optional key.
        if value is None:
            return None

        elif isinstance(temp, dict):
            result = JsonData._deserialize_dict(temp, value, [], path)
            return result

        # list,
        elif isinstance(temp, list):
            assert value.__class__ is list
            l = []
            if len(temp) == 0:
                l=value
            elif len(temp) == 1:
                for ival, v in enumerate(value):
                    value = JsonData._deserialize_item(temp[0], v, path + [str(ival)])
                    if value is not None:
                        l.append(value)
            else:
                # print("Warning: Overwriting default list content:\n {}\n path:\n {}.".format(temp, path))
                l=value
            return l

        # tuple,
        elif isinstance(temp, tuple):
            assert isinstance(value, (list, tuple)), "Expecting list, get class: {}\,path: {}".format(value.__class__, path)
            assert len(temp) == len(value), "Length of tuple do not match: {} != {}".format(len(temp), len(value))
            l = []
            for i, tmp, val in zip(range(len(value)), temp, value):
                l.append(JsonData._deserialize_item(tmp, val, path + [str(i)]))
            return tuple(l)

        # ClassFactory - class given by '__class__' key.
        elif isinstance(temp, ClassFactory):
            return temp.make_instance(value, path)

        # JsonData default value, keep the type.
        elif isinstance(temp, JsonData):
            return ClassFactory( [temp.__class__] ).make_instance(value, path)

        # other scalar types
        else:
            if inspect.isclass(value):
                return None
            # only temp type matters
            if not inspect.isclass(temp):
                temp = temp.__class__

            if issubclass(temp, IntEnum):
                if value.__class__ is str:
                    return temp[value]
                elif value.__class__ is int:
                    return temp(value)
                elif isinstance(value, temp):
                    return value
                else:
                    assert False, "{} is not value of IntEnum: {}\npath: {}".format(value, temp, path)

            else:
                try:
                    filled_template = temp(value)
                except:
                    raise Exception("Can not convert value {} to type {}.\npath: {}".format(value, temp, path))
                return filled_template


    def serialize(self):
        """
        Serialize the object.
        :return:
        """
        return self._get_dict()


    # def _is_attr_serialized(self, attr):
    #     """
    #     Return True if attribute is serialized.
    #     :param attr: attribute
    #     :return:
    #     """
    #     if self._serialized_attr is None:
    #         if attr[0] != "_":
    #             return True
    #     else:
    #         if attr in self._serialized_attr:
    #             return True
    #     return False


    def _get_dict(self):
        """Return dict for serialization."""
        d = {"__class__": self.__class__.__name__}
        for k, v in self.__dict__.items():
            if k not in self._not_serialized_attrs_ and not isinstance(v, ClassFactory):
                d[k] = JsonData._serialize_object(v)
        return d

    @staticmethod
    def _serialize_object(obj):
        """Prepare object for serialization."""
        if isinstance(obj, JsonData):
            return obj._get_dict()
        elif isinstance(obj, IntEnum):
            return obj.name
        elif isinstance(obj, dict):
            d = {}
            for k, v in obj.items():
                d[k] = JsonData._serialize_object(v)
            return d
        elif isinstance(obj, list) or isinstance(obj, tuple):
            l = []
            for v in obj:
                l.append(JsonData._serialize_object(v))
            return l
        else:
            return obj

    # @staticmethod
    # def make_instance(config):
    #     """
    #     Make instance from config dict.
    #     Dict must contain item "__class__" with name of desired class.
    #     :param config:
    #     :return:
    #     """
    #     if "__class__" not in config:
    #         return None
    #
    #     # find class by name
    #     cn = config["__class__"]
    #     if cn in locals():
    #         c = locals()[cn]
    #     elif cn in globals():
    #         c = globals()[cn]
    #     else:
    #         return None
    #
    #     # instantiate class
    #     d = config.copy()
    #     del d["__class__"]
    #     return c(d)
