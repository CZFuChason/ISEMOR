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
    
Given the features, we randomly selected 20% samples of training set for random masking with numpy.random.uniform() function.

For your convinient, you can download the dataset from here (available soon)

Then, split the data for training and testing set (leave-one-session-out):
    
    python get_loso.py (will be uploaded later)

Put the pickled features to 'data' folder.
 
To start training, simply run

    python train.py
    
Since the model involves the adversarial traing, you may have to be patient to train the model serveral times to get the best result.


## Konwn issues
1. This demo code does not contain early stop fucntion, you need to write the function by yourself.
will be given soon
