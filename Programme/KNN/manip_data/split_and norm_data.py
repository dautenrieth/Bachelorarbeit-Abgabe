
import pickle
import numpy as np

# max vol -> 6.85
# max kg -> 3
# max a -> 75

a_file = open("data.pkl", "rb")
data = pickle.load(a_file)
a_file.close()

data_input = []
data_output = []
count = 0

for i in data:
    norm_val = [elem/6.85 for elem in data[i]['Volumenverlauf'][0]]
    data_input.append(norm_val)
    data_input[count].append(data[i]['Parameter']['Koerpergroeße']/2)
    data_input[count].append(data[i]['Parameter']['Alter']/75)
    data_input[count].append(data[i]['Parameter']['Geschlecht'])
    count += 1
    

for i in data:
    data_output.extend(data[i]['Output'])

a_file = open("input_data.pkl", "wb")
pickle.dump(data_input, a_file)
a_file.close()

a_file = open("output_data.pkl", "wb")
pickle.dump(data_output, a_file)
a_file.close()

print('finished')