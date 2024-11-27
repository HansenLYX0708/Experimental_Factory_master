import numpy as np

if __name__ == '__main__':
    a = np.array([[[1, 2, 3, 4],[5,6,7,8],[9,10,11,12]],
                  [[13,14,15,16],[17,18,19,20],[21,22,23,100]]])
    print(a.shape)

    a_ave_0 = np.mean(a, axis=0)
    a_ave_0_2 = np.mean(a_ave_0, axis=1)

    a_ave_2 = np.mean(a, axis=2)
    a_ave_2_0 = np.mean(a_ave_2, axis=0 )


    print('end')
