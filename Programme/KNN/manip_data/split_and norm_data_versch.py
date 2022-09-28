
import pickle
import numpy as np

# max vol -> 6.85
# max kg -> 2
# max a -> 75

a_file = open("data_versch.pkl", "rb")
data = pickle.load(a_file)
a_file.close()

data_input_train = []
data_output_train = []
data_input_test = []
data_output_test = []
count = 0
count_train = 0
count_test = 0

for i in data:
    for x in range(0, len(data[i]['Volumenverlauf'])):
        if ((count % 10 == 0 or count % 11 == 0 or count % 12 == 0 or count % 13 == 0 or count % 14 == 0 ) and count != 0):
            norm_val = [elem/6.85 for elem in data[i]['Volumenverlauf'][x]]
            data_input_test.append(norm_val)
            data_input_test[count_test].append(data[i]['Parameter']['Koerpergroeße']/2)
            data_input_test[count_test].append(data[i]['Parameter']['Alter']/75)
            data_input_test[count_test].append(data[i]['Parameter']['Geschlecht'])
            data_output_test.extend(data[i]['Output'])
            count_test += 1
        else:
            norm_val = [elem/6.85 for elem in data[i]['Volumenverlauf'][x]]
            data_input_train.append(norm_val)
            data_input_train[count_train].append(data[i]['Parameter']['Koerpergroeße']/2)
            data_input_train[count_train].append(data[i]['Parameter']['Alter']/75)
            data_input_train[count_train].append(data[i]['Parameter']['Geschlecht'])
            data_output_train.extend(data[i]['Output'])
            count_train += 1
    count += 1
    if count % 25 == 0:
        print(count)


a_file = open("input_data_versch_test.pkl", "wb")
pickle.dump(data_input_test, a_file)
a_file.close()
a_file = open("input_data_versch_train.pkl", "wb")
pickle.dump(data_input_train, a_file)
a_file.close()

a_file = open("output_data_versch_train.pkl", "wb")
pickle.dump(data_output_train, a_file)
a_file.close()

a_file = open("output_data_versch_test.pkl", "wb")
pickle.dump(data_output_test, a_file)
a_file.close()

print('finished')