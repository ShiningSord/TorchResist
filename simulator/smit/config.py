
class SimulationConfigs:
    def __init__(self, 
                 LAYERPATH : str = None, 
                 layer : int = 1, 
                 pixel : int = 28) -> None:
        layerinfo = {}
        if LAYERPATH == None:
            raise ValueError("please input the gds file path")
        else:
            layerinfo["path"]=LAYERPATH
        
        layerinfo["main"]={"layer": layer, "datatype":0} #setting same with gds 

        bright={"trans":1, "phase":0} #setting save with real mask
        dark={"trans": 0.00, "phase":180}
        self.pixel = pixel
        transmit={}
        transmit["field"]=dark
        transmit["main"]=bright   

        self.all_setting = {}
        self.all_setting["layerinfo"] = layerinfo
        self.all_setting["transmit"] = transmit


        self.allsettinggrad={}
        self.allsettinggrad["computemode"]="fast" #normal or fast

        self.all_setting["pixel"] = pixel