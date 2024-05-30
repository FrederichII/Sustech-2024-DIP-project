import numpy as np
import matplotlib.pyplot as plt


def median(s,kernel):
    paddingSize = int(kernel / 2)
    s = np.array(s)
    s_padded = np.array(s+2*paddingSize)
    result = np.zeros(s_padded.shape)
    for i in range(paddingSize):
        s_padded[i] = 0
    for i in range(len(s_padded)-1-paddingSize,len(s_padded)):
        s_padded[i] = 0
    for i in range(paddingSize,len(s_padded)-2-paddingSize):
        s_padded[i] = s[i-paddingSize]
        
    for i in range(paddingSize,len(s_padded)-2-paddingSize):
        region = s_padded[i-paddingSize:i+paddingSize+1]
        result[i] = np.median(region)
    
    result = result[paddingSize:len(s_padded)-2-paddingSize]
    return result

if __name__ == '__main__':
    t = np.linspace(0,2*np.pi,256)
    s = np.sin(t)
    for i in range(len(s)):
        s[i] = s[i] + 0.2*(np.random.rand()-0.5)
    
    s_median = median(s,5)
    plt.figure()
    plt.plot(s)
    plt.figure()
    plt.plot(s_median)
    plt.show()