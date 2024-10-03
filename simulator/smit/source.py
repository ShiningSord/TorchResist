try:
    import sourcepy
except Exception as e:
    print("We do not find sourcepy package.")
    pass

class Source:
    sigmain=0.7
    sigmaout=0.9
    angle=30
    rot=45
    sourcetype="quasar" #only "dipole dipolex dipoley quasar quad annular" can be setted
    symmetry="D4" #only "D2 D4" can be setted
    
    def __init__(self, 
                 path : str) -> None:
        self.path = path
        
    def run(self):
        sourcepy.getsource(self.sigmain, 
                           self.sigmaout, 
                           self.angle, 
                           self.rot, 
                           self.sourcetype, 
                           self.symmetry)
        sourcepy.showsource(f"{self.path}/source.src")