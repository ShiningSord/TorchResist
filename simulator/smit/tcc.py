
try:
    import tccpy
except Exception as e:
    print("We do not find tccpy package.")
    pass


class Tcc:
    def __init__(self, 
                 mode : str, 
                 pixel=28, 
                 TCCSAVEPATH=None, 
                 SOURCEFILE=None) -> None:
        self.pixel = pixel
        
        assert mode in ["normal", "inner", "outer"]
        self.mode = mode

        if TCCSAVEPATH == None:
            raise ValueError("please input the tcc file path")
        else:
            self.TCCSAVEPATH = f"{TCCSAVEPATH}/{self.mode}_tcc"

        if SOURCEFILE == None:
            raise ValueError("please input the source.src file path")
        else:
            self.SOURCEFILE = f"{SOURCEFILE}/source.src"

        source={}
        source["file"]=self.SOURCEFILE
        source["poltype"]="TE" #only "TM X Y XY" can be setted
        source["symmetry"]="D4" #only "D2 D4" can be setted
        source["grid"]=20
        source["wavelength"]=193
        source["na"]=1.35

        film=[]
        film.append({'n':1.69,'k':0.048,'t':90}) #setting save with resist, if not reist, please not append, however you can not delete film=[]

        substract=[]
        substract.append({'n':0.883,'k':2.777})#setting save with substract, if not substract, please not append, however you can not delete substract=[] 

        parameters={}
        parameters["n_m"]=1.4366 #1.44 water
        parameters["redunction"]=4
        parameters["tcc_grid"]=10
        parameters["savepath"]=self.TCCSAVEPATH
        parameters["pixelsize"]=pixel

        condition={}
        if self.mode == "inner":
            self.focus = -30
            # condition["focus"]=-30 #posivite in resist
        elif self.mode == "outer":
            self.focus = 30
            # condition["focus"]=30 #posivite in resist
        else:
            self.focus = 0
            # condition["focus"]=0 #posivite in resist
        condition["focus"]=self.focus

        condition["image_z"]=0 #posivite in resist

        self.all_setting={}
        self.all_setting["source"]=source
        self.all_setting["film"]=film
        self.all_setting["substract"]=substract
        self.all_setting["parameters"]=parameters
        self.all_setting["condition"]=condition
       
    def computeTCC(self):
        tccpy.computeoptic(self.all_setting)