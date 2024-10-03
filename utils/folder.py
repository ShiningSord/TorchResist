import os
import fuILT
from pathlib import Path
import hashlib

workspace = ".fuILT"

def getFuILTPath():
    return fuILT.__path__[0]

def getFuILTBenchmarkPath(self):
    return getFuILTPath() + "/../benchmarks"

def getHomePath():
    return str(Path.home())

def buildTheWorkSpace():
    path = getWorkspacePath()
    if not os.path.exists(path):
        os.makedirs(path)
        
def getWorkspacePath():
    path =  getHomePath() + "/" + workspace
    assert os.path.exists(path)
    return path

def buildFolderInWorkspace(name : str):
    path = getWorkspacePath() + "/" + name
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def getFolderInWorkspace(name : str):
    path = getWorkspacePath() + "/" + name
    assert os.path.exists(path)
    return path
        
def buildTempFolderInWorkspace(path : str):
    path = path + "/temp"
    if not os.path.exists(path):
        os.makedirs(path)
        
    return path

def getConfigPath():
    path = getFuILTPath() + "/../config"
    return path

def nameHashing(layout : str, layer : int, pixel : int):
    md5 = hashlib.md5()
    md5.update(bytes(f"{layout}_{layer}_{pixel}", 'utf-8'))
    return md5.hexdigest()
    
def removeTempFolder(path):
    import shutil
    assert path[-4:] == "temp"
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)



