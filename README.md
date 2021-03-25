# MAEC & ISO-MAEC

Data preprocessing: follow the [opensmile document](https://audeering.github.io/opensmile/get-started.html#extracting-features-for-emotion-recognition)

After extracting features with opensmile toolkit, use following function to re-align features

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
    
    
Put the pickled features to 'data' folder.
or you can directly download and unzip our [preprocessed data (leave-one-session-out)](https://drive.google.com/file/d/1SjfmzuZEzzd0pVM-_zR03id1UX8aVqbp/view?usp=sharing)
(available soon)
 
To start training, simply run

    python train.py
    
Since the model involving the adversarial traing, you may be patient to train the model serveral times to get the best result.
