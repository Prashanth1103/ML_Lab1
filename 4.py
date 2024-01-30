import pandas as pd

df = pd.read_csv(r'C:\Users\prash\Downloads\enjoysport.csv')
print("Data is : \n\n", df)
print("\n", "*"*100)
print("Shape of dataset is : ", df.shape, "\n")

n = 6
a = []
for i in range(len(df)):
    a.append(df.iloc[i].tolist())

hypo = ['0'] * n
for i in range(0, n):
    hypo[i] = a[0][i]

print("\nMost Specific Hypothesis is : ", hypo)

for i in range(len(a)):
    if a[i][n] == "yes":
        for j in range(0, n):
            if a[i][j] != hypo[j]:
                hypo[j] = '?'
    else:
        hypo[j] = a[i][j]
    print(f"\nFor Training instance {i+1}, The hypothesis is : \n{hypo}\n")

print(f"The maximally specific hypothesis is : \n{hypo}")

