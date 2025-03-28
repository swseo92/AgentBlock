import glob
from agentblock.schema.tools import validate_yaml


list_fpath_yaml = glob.glob("./*/*.yaml")
print(list_fpath_yaml)

for fpath_yaml in list_fpath_yaml:
    print(fpath_yaml)
    validate_yaml(fpath_yaml)
