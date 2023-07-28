import os
import glob

completion_list = glob.glob("*f.ply")

for ll in completion_list:
    os.rename(ll, ll.split(".")[0][:-2]+ll.split(".")[1])

file_list = glob.glob()

def save_list_to_txt(lst, filename):
    with open(filename, 'w') as file:
        file.write('\n'.join(lst))


save_list_to_txt(file_list, "output.txt")
