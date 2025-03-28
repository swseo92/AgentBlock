import glob
from agentblock.schema.tools import validate_yaml


list_fpath_yaml = glob.glob("**/*.yaml", recursive=True)

for fpath_yaml in list_fpath_yaml:
    try:
        validate_yaml(fpath_yaml)
    except Exception as e:
        print(fpath_yaml, "is not valid yaml", e)
