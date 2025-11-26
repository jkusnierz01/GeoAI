import os

def get_prefix(filename):
    """
    Extracts class/group prefix from filename.
    For files like 'abc_res7.geojson' → return 'abc'.
    """
    base = os.path.basename(filename)
    if "_res" in base:
        return base.split("_res")[0]
    return base.split(".")[0]
