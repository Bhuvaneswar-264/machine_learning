import numpy as np
def compute_dot_product(v1,v2):
    return sum(a*b for a,b in zip(v1,v2))
def compute_euclidean_norm(vector):
    return sum(v*v for v in vector)**0.5
dataframe = pd.read_csv("BERT_embedding.csv")
label_column = "label"
features = dataframe.drop((columns=[label_column]).values
A=features[0]
B=features[1]
dot=compute_dot_product(A,B)
norm=compute_euclidean_norm(A)
numpy_dot=np.dot(A,B)
numpy_norm=np.linalg.norm(A)
print("Dot Product :",dot)
print("Dot Product :",numpy_dot)
print("Euclidean Norm:",own_norm)
print("Euclidean Norm:",numpy.norm)                          
