import hashlib
import json

def md5file(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
        
    return hasher.hexdigest()

def md5array(data):
    return hashlib.md5(json.dumps(data.tolist(), sort_keys=True).encode('utf-8')).hexdigest()