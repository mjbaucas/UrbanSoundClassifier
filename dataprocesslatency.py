


latency_file = open("hybrid_latency_12.txt", "r")

latency_dict = {}
for line in latency_file.readlines()[180:]:
    split_text = line.strip().split(" ")
    if split_text[1] not in latency_dict:
        latency_dict[split_text[1]] = []
    latency_dict[split_text[1]].append(float(split_text[2]))

latency_dict_ave = {}
for key, value in latency_dict.items():
    counter = 0
    if key not in latency_dict_ave:
        latency_dict_ave[key] = 0
    for x in range(len(value)):
        if(x%2 == 0):
            if (value[x+1] - value[x]) < 1 and (value[x+1] - value[x]) > 0:
                latency_dict_ave[key]+= (value[x+1] - value[x]) 
                counter+=1
    latency_dict_ave[key] = latency_dict_ave[key]/counter

#print(latency_dict_ave)

average = 0
counter = 0
for key, value in latency_dict_ave.items():
    average += value
    counter += 1

average = average/counter
print(average)