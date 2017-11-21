import os
path = "/home/experimentality/Documents/Inteligencia/final_inteligencia/Allowed/"
files = os.listdir(path)
i = 1

for file in files:

    orig_name = file
    os.rename(os.path.join(path, file), os.path.join(path, '1.alejo' + str(i) + '.jpg'))
    i = i + 1
