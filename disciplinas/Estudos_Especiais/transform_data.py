
import numpy as np
import re
from os import listdir
from os.path import isfile, join

def transform_data(): 
    data_training_files = [f for f in listdir('./raman-spectroscopy-of-candida-fungo/original_data') if isfile(join('./raman-spectroscopy-of-candida-fungo/original_data', f))]

    data = []

    header = 'Albicans/Glabrata'

    for i in range(1024):
        header += ',Feature %d' % i

    for f in data_training_files:        
        x = np.genfromtxt('./raman-spectroscopy-of-candida-fungo/original_data/' + f)[:,1]
        data.append(np.concatenate((np.array([float(int("Albicans" in f))]), x))) 
    
    return data, header

def main():
    data, header = transform_data()
    np.savetxt("./raman-spectroscopy-of-candida-fungo/candida.csv", data, delimiter=",", header=header)
  

if __name__ == "__main__":
    main()