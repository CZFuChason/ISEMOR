# MAEC

After extracting features with opensmile toolkit, we used following function to re-align features

    def getFeature(File_path):
        File = open(File_path,"r")
        Data = File.readlines()
    
        Mfcc = np.arange(4,16)
        Mfcc_de = np.arange(30,42)
        Mfcc_de_de = np.arange(56,68)
    
        Lound = np.array([3])
        Lound_de = np.array([29])
        Lound_de_de = np.array([55])
    
        F0 = np.array([26])
        F0_de = np.array([52])
    
        Voice_Pro = np.array([25])
        Voice_Pro_de = np.array([51])
    
        Zcr = np.array([24])
        Zcr_de = np.array([50])
    
        Index = np.concatenate((F0,F0_de,Lound,Lound_de,Lound_de_de,Voice_Pro,Voice_Pro_de,Zcr,Zcr_de,Mfcc,Mfcc_de,Mfcc_de_de))
    
        All_Feature = []
        for data in Data[86:]:
            feature = np.array(np.array(data.split(","))[Index],dtype=float)
            All_Feature.append(feature)
        
        All_Feature = np.asarray(All_Feature)
    
        return All_Feature
    
    
    
Since the model involving the adversarial traing, you may be patient to train the model serveral times to get the best result.
