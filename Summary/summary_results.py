import pickle
import numpy as np

# with open("./output/cifar_inception2.pkl", 'rb') as f:
#     dat = pickle.load(f)
#     is_dict = dict({})
#     for item in dat:
#         allis = dat[item]
#         allis = [x[0] for x in allis]
#         is_dict[item] = np.array(allis)
#         print(item, np.max(is_dict[item]),
#               np.where(is_dict[item] == np.max(is_dict[item])))

# with open("./output/cifar_inception5.pkl", 'rb') as f:
#     dat = pickle.load(f)
#     is_dict = dict({})
#     for item in dat:
#         allis = dat[item]
#         allis = [x[0] for x in allis]
#         is_dict[item] = np.array(allis)
#         print(item, np.max(is_dict[item]),
#               np.where(is_dict[item] == np.max(is_dict[item])))

# with open("./output/cifar_inception10.pkl", 'rb') as f:
#     dat = pickle.load(f)
#     is_dict = dict({})
#     for item in dat:
#         allis = dat[item]
#         allis = [x[0] for x in allis]
#         is_dict[item] = np.array(allis)
#         print(item, np.max(is_dict[item]),
#               np.where(is_dict[item] == np.max(is_dict[item])))

# with open("./output/cifar_inceptionhinge.pkl", 'rb') as f:
#     dat = pickle.load(f)
#     is_dict = dict({})
#     for item in dat:
#         allis = dat[item]
#         allis = [x[0] for x in allis]
#         is_dict[item] = np.array(allis)
#         print(item, np.max(is_dict[item]),
#               np.where(is_dict[item] == np.max(is_dict[item])))

with open("./output/cifar_inception_plot.pkl", 'rb') as f:
    dat = pickle.load(f)
    is_dict = dict({})
    for item in dat:
        allis = dat[item]
        allis = [x[0] for x in allis]
        is_dict[item] = np.array(allis)
        print(item, np.max(is_dict[item]),
              np.where(is_dict[item] == np.max(is_dict[item])))
