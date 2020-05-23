import sys
sys.path.insert(1, './')
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]