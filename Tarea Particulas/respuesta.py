import numpy as np
import matplotlib.pyplot as plt

preg = input("Pregunta a responer: ") 

fig = plt.figure()

arrayy = np.array([0 , 0, 0, 0, 1, 1, 2,2,2,2,3,4,4,4,4])
arrayx = np.array([0,1,2,4,2,4,0,1,2,4,0,0,1,2,4])

arrayx2 = np.array([0, 3, 5,6,7,0,2,3,5,7,0,1,3,5,7,0,3,5,6,7])
arrayy2 = np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])

arrayx3 = np.array([0,1,2,4,6,0,2,4,6,0,1,2,4,5,6,0])
arrayy3 = np.array([0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,3])

if int(preg) == 1:
    plt.scatter(arrayx3, arrayy3)
    plt.show()
elif int(preg) == 2:
    print("nose dejen ver")

elif int(preg) == 3:
    plt.scatter(arrayx2,arrayy2)
    plt.show()

