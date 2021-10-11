import numpy as np
def calculateCross(A,B,C,D):
    lineAC = C - A
    lineAD = D - A
    lineBC = C - B 
    lineBD = D - B
    if (min(A[0] ,B[0])<= max(C[0],D[0])) and (min(C[0] ,D[0])<= max(A[0],B[0])) and (min(A[1] ,B[1])<= max(C[1],D[1])) and (min(C[1] ,D[1])<= max(A[1],B[1])):
        def vectorProduct(vector1,vector2):
            return vector1[0] * vector2[1] - vector2[0] * vector1[1]
        return (vectorProduct(lineAC,lineAD)*vectorProduct(lineBC,lineBD)<=0) and (vectorProduct(lineAC,lineBC)*vectorProduct(lineAD,lineBD)<=0)
    else:
        return False

if __name__ == '__main__':

    A = np.array([0,1])
    B = np.array([1,1])
    C = np.array([1,0])
    D = np.array([2,2])
    print(calculateCross(A,B,C,D))
    