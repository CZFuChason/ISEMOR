import numpy as np
import pickle as pkl
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence



def multi_instances(samples, width, stride):
    mi_samples = []
    l = len(samples)
    for si in range(l):
        print('%d/%d'%(si, l), end='\r')
        s = samples[si]
        s_mi = []
        stfla = False
        for i in range(0, len(s), stride):
            try:
                s = s.tolist()
            except:
                pass
            tmp = s[i:i+width]
            while len(tmp)<width:
                tmp.append(np.zeros((45)))
                stfla = True
            s_mi.append(np.asarray(tmp))
        
            if stfla:
                break
        
        mi_samples.append(np.asarray(s_mi))
    mi_samples = sequence.pad_sequences(mi_samples, maxlen=70, 
                                        dtype='float32', 
                                        padding='post', value=0.)
    print(mi_samples.shape)
    return np.asarray(mi_samples)  

def get_data():
    
    tra_smls = pkl.load(open('../data/tra_smls.p', 'rb'))[:-10]
    tra_intlter = pkl.load(open('../data/tra_intlter.p', 'rb'))[:-10]
    tra_pre_self = pkl.load(open('../data/tra_pre_self.p', 'rb'))[:-10]

    tra_emos = pkl.load(open('../data/tra_emos.p', 'rb'))[:-10]
    tra_gends = pkl.load(open('../data/tra_gends.p', 'rb'))[:-10]


    tes_smls = pkl.load(open('../data/tes_smls.p', 'rb'))
    tes_intlter = pkl.load(open('../data/tes_intlter.p', 'rb'))
    tes_pre_self = pkl.load(open('../data/tes_pre_self.p', 'rb'))

    tes_emos = pkl.load(open('../data/tes_emos.p', 'rb'))
    tes_gends = pkl.load(open('../data/tes_gends.p', 'rb'))

    w = 50
    stride = 10
    print('width=%d, stride=%d'%(w, stride))
    mi_tra_specs_ = multi_instances(tra_smls, w, stride)
    mi_tra_intlter_ = multi_instances(tra_intlter, w, stride)
    mi_tra_pre_self_ = multi_instances(tra_pre_self, w, stride)

    mi_tes_specs_ = multi_instances(tes_smls, w, stride)
    mi_tes_intlter_ = multi_instances(tes_intlter, w, stride)
    mi_tes_pre_self_ = multi_instances(tes_pre_self, w, stride)
    
    tra_emos_ = six2four(tra_emos)
    tes_emos_ = six2four(tes_emos)
    tra_gends_ = tra_gends
    tes_gends_ = tes_gends
    
    return [mi_tra_specs_, mi_tra_intlter_, mi_tra_pre_self_, tra_emos_, tra_gends_,
            mi_tes_specs_, mi_tes_intlter_, mi_tes_pre_self_, tes_emos_, tes_gends_]




