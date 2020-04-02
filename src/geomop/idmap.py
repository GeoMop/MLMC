"""
A dictionary that automaticaly assign unique IDs to appended objects
- IDs are set to the objects
- Objects can be compared by IDs.
TODO: Make auxiliary class for producing IDs and allow
several IdMaps to source from common ID source
"""
class IdObject:

    def __init__(self):
        self.attr = None

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


class IdSource:
    pass

class IdMap(dict):
    #id = 0
    def __init__(self, id_source=IdSource()):
        self._next_id = -1
        super().__init__()


    def get_new_id(self):
        self._next_id += 1
        return self._next_id

    def append(self, obj, id = None):
        if id is None:
            id = self.get_new_id()
        else:
            self._next_id = max(self._next_id, id)
        obj.id = id
        self[obj.id] = obj
        return obj
