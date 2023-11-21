"""
this function is from the following stackoverflow post:
https://stackoverflow.com/questions/50008296/facebook-json-badly-encoded
It converts the latin-1 encoded strings to utf-8 encoded strings due to Facebook's bad encoding.
It iteratively does this for all strings in the json object.
"""
def unicode_converter(obj: dict):
    if isinstance(obj, str):
        return obj.encode('latin_1').decode('utf-8')

    if isinstance(obj, list):
        return [unicode_converter(o) for o in obj]

    if isinstance(obj, dict):
        return {key: unicode_converter(item) for key, item in obj.items()}

    return obj