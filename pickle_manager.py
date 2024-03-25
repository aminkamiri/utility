import pickle
def save_as_pickle(obj,filename):
    file = open(filename, 'wb')
    pickle.dump(obj, file)
    file.close()

def open_a_pickle(filename):
    file = open(filename, 'rb')
    obj=pickle.load(file)
    file.close()
    return obj
