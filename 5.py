import numpy as np
import pandas as pd

data = pd.read_csv(r'C:\users\prash\downloads\PlayTennis.csv')
data = pd.DataFrame(data)

concepts = np.array(data.iloc[:, 0:-1])
print("\n", concepts, "\n")
target = np.array(data.iloc[:, -1])
print("\n", target, "\n")

def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("Initialization of specific_h and general_h")
    print(specific_h)
    general_h = [['?' for i in range(len(specific_h))] for i in range(len(specific_h))]
    print(general_h)
    for i, h in enumerate(concepts):
        print(f"Instance {i+1} is {h}")
        if target[i] == 'Yes':
            print("Instance is positive")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
                print(specific_h)
        print(specific_h)
        if target[i] == 'No':
            print("Instance is negative")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
        print("Steps of candidate elimination algorithm", i+1)
        print(specific_h)
        print(general_h)
        print("\n")
    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?']]
    return specific_h, general_h
    
sfinal, gfinal = learn(concepts, target)
print("Final Specific_h : ", sfinal)
print("\nFinal General_h : ", gfinal)

