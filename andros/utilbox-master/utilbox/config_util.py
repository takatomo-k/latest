import json
import importlib
import re

__all__ = ['ConfigParser']
class ConfigParser(object) :
    """
    JSON-based config parser
    """

    @staticmethod
    def list_parser(obj, n=1) :
        obj = json.loads(obj) if isinstance(obj, str) else obj 
        obj = [obj for _ in range(n)] if not isinstance(obj, list) else obj 
        while len(obj) < n :
            obj = obj + [obj[-1]]
        return obj

    @staticmethod
    def item_parser(obj) :
        return json.loads(obj) if isinstance(obj, str) else obj
