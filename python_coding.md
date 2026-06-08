# Default rules for Python coding

These rules specify the coding practices and style for the projects of the 
Multiphysics Software Group at Technical University of Liberec.
There are three sections:
- common rules
- rules for research project (single purpose code, used internaly)
- rules for production projects (libraries, SW intended for general usage)

## Common rules

### Common sense rules
- NEVER resolve test errors by try blocks
- do not use "guess" default values, only obvious defaults
- Be defensive, with strong checks, but only for the user input data (config files usualy)
  That means error inputs must raise early. Therefore only check for existing keys in input dicts
  if these will be required down in a long calculation. Otherwise just let KeyError do the job, do not catch it.
- Do not use other config keys in the case of a KeyError, just throw early.
- Use asserts for consistency checks.´
- NEVER add runtime fallbacks or import shims to compensate for a broken or incomplete environment.
  If a required dependency or tool is missing, report the environment problem plainly and fix the environment or tests around it, but do not implement code workarounds.
- Optional dependencies may be imported lazily through a clear helper function at the feature boundary.
  The helper must fail with a direct message naming the missing optional dependency and the feature that needs it.
- NEVER write "self explanatory" into doc comments. Even if the variable/key could be obvious explain it in other words also with bit of context to avoid confusion.

### Best code, is no code!** 
  - Suggest refactoring (even breaking backward compatibility) that would reduce the size or complexity of the code base. 
  - Avoid code duplicities.
  - prefere high level code: numpy, pandas, xarray instead loops and native python sturctures (lists, dicts)
    For the sake of both speed and less code
  - Avoid branching and nested branching in particular. Use polymophism and duck typing o
  - Avoid introducing simple module or class functions that are used only once.
    If a small helper is needed only inside one expression or method, prefer a
    local `def` or `lambda`. Use a private `_*` helper only when it is reused or
    when keeping it separate clearly simplifies the surrounding code.
  - Do not constrain the implementation by the tests at early stages of implementation. 
    Rather be open to radical rafactoring and rewrite of the tests. 
  
### Antipaterns
  - Do not index tuple. To access tuple items, assign it to named local variables. That documents tuple components.
    That slso makes use of large tuples (len > 3) uncomfortable pushing towards proper dataclass.
  
### Functional style  
  - Prefere functional style (pure functions)
  - idealy do not change objects after construction, methods only do calculations, reading the data in the class (and other passed arguments)

### Documentation
  - Create doc strings for created public classes, functions, methods. 
  - Document goals of each pytest testfunction
  
### Prefered libraries
- use logging
- Use logging for debug outputs.
- use pathlib
- use attrs for dataclasses
- use attrs staticmethod/classmethod technique to construct from other data then is stored in the dataclass  


## Research project rules
Here will be rules applied to reseach codes.
Less defensive, no backward compatibility,  structure: input -> output

## Production code rules
More complex, more defensive, backward compatibility.
