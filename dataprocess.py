from matplotlib import pyplot
import numpy
import csv

def get_times(data_file):
    times = {}
    for line in data_file.readlines():
        line = line.strip()

        if line != " " or line != None:
            first_split = line.split(": ")
            header = first_split[0]
            value = first_split[1]
            second_split = header.split(" ")
            label = second_split[0]
            number = int(second_split[1])

            if not number in times:
                times[number] = {}
            times[number][label] = value            
    
    return times

def organize_data(time_dict):
    record_list = []
    classifier_list = []
    extract_list = []
    time_list = []

    time_counter = 0
    override_check = 0
    time_list.append(0)
    for key, value in time_dict.items():
        if 'Middle1' in value:
            time_dict[key] = [float(value['End']) - (float(value['Middle2']) + 4), (float(value['Middle2']) + 4) - float(value['Middle1']) , float(value['Middle1']) - float(value['Start'])]
            record_list.append(time_dict[key][2])
            extract_list.append(time_dict[key][1])
            classifier_list.append(time_dict[key][0])
            time_counter += time_dict[key][2]
            time_list.append(time_counter)
            time_counter += time_dict[key][1]
            time_list.append(time_counter)
            time_counter += time_dict[key][0]
            time_list.append(time_counter) 
            override_check = 1   
        else:
            time_dict[key] = [float(value['End']) - float(value['Middle']), float(value['Middle']) - float(value['Start'])]
            record_list.append(time_dict[key][1])
            classifier_list.append(time_dict[key][0])
            time_counter += time_dict[key][1]
            time_list.append(time_counter)
            time_counter += time_dict[key][0]
            time_list.append(time_counter)
    
    if len(extract_list) > 10:
        extract_list = extract_list[10:]

    if override_check == 1:
        time_list = time_list[30:]
    else:
        time_list = time_list[20:]
    
    return record_list[10:], classifier_list[10:], extract_list, time_list

def get_power_list(time_list, csv_file):
    power_dict = {}
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            power_dict[float(row['Time (s)'])] = [float(row['USB Avg Current (mA)']), float(row['USB Avg Power (mW)'])]     
    
    power_list = []
    lower_bound = 0
    higher_bound = 0
    for time_value in time_list:
        lower_bound = higher_bound
        higher_bound = time_value
        
        ave_value = 0
        counter = 0
        for key, value in power_dict.items():
            if key <= higher_bound and key >= lower_bound:
                ave_value += value[1]
                counter += 1
        
        if counter != 0:
            ave_value = ave_value/counter
        power_list.append([higher_bound, lower_bound, ave_value])
    
    return power_list

def organize_power_list(power_list, override):
    power_record_list = []
    power_extract_list = []    
    power_classifier_list = []
    switch = 0
    for power_value in power_list:
        if switch == 0:
            power_record_list.append(power_value[2])
            if override == 1:
                switch = 2
            else:
                switch = 1
        elif switch == 1:
            power_extract_list.append(power_value[2])
            switch = 2
        else:
            power_classifier_list.append(power_value[2])
            switch = 0
    
    return power_record_list, power_extract_list, power_classifier_list

def add_results(local_record, local_classify, server_record, server_classify, hybrid_record, hybrid_extract, hybrid_classify):
    # Add both 
    local_temp = []
    server_temp = []
    hybrid_temp = []
    for x in range(0,20):
        local_temp.append((local_record[x] + local_classify[x])/2)
        server_temp.append((server_record[x] + server_classify[x])/2)
        hybrid_temp.append((hybrid_record[x] + hybrid_extract[x] + hybrid_classify[x])/3)
        
    #pyplot.scatter(range(1,21), local_temp, 20, label='Configuration A')
    #pyplot.scatter(range(1,21), server_temp, 20, label='Configuration B')
    pyplot.bar(range(1,21), local_temp, width=0.3,label="Local", align="center")
    pyplot.bar(numpy.arange(1.3, 21.3, 1.0), server_temp, width=0.3,label="Server", align="center")
    #pyplot.scatter(range(1,21), hybrid_temp, 20, label='Configuration C')
    pyplot.legend(loc='best')
    pyplot.ylabel('Average Power (mW)')
    pyplot.xlabel('Program Iterations')
    pyplot.xticks(list(range(0,21))[::5])
    pyplot.ylim([1800, 1900])
    pyplot.savefig('power_noise.png')
    pyplot.close()

if __name__ == "__main__":
    data_file_local = open("edgetime_model.txt", "r")
    data_file_server = open("cloud_time_no_wifi_toggle.txt", "r")
    data_file_hybrid = open("hybrid_time.txt", "r")

    times_local = get_times(data_file_local)
    times_server = get_times(data_file_server)
    times_hybrid = get_times(data_file_hybrid)


    record_list_local, classifier_list_local, [], time_list_local= organize_data(times_local)
    record_list_server, classifier_list_server, [], time_list_server = organize_data(times_server)
    record_list_hybrid, classifier_list_hybrid, extract_list_hybrid, time_list_hybrid = organize_data(times_hybrid)

    #print(time_list_local)
    #print(time_list_server)
    
    pyplot.scatter(range(1,21), record_list_local, 20, label='Recording A')
    pyplot.scatter(range(1,21), classifier_list_local, 20, label='Classifying A')
    pyplot.scatter(range(1,21), record_list_server, 20, label='Recording B')
    pyplot.scatter(range(1,21), classifier_list_server, 20, label='Classifying B')
    #pyplot.scatter(range(1,21), record_list_hybrid, 20, label='Recording (Hybrid)')
    #pyplot.scatter(range(1,21), extract_list_hybrid, 20, label='Extracting (Hybrid)')
    #pyplot.scatter(range(1,21), classifier_list_hybrid, 20, label='Classifying (Hybrid)')
    pyplot.legend(loc='best')
    pyplot.ylabel('Time (s)')
    pyplot.xlabel('Program Iterations')
    pyplot.xticks(list(range(0,21))[::5])
    pyplot.savefig('runtime_locserv.png')
    pyplot.close()

    print(sum(record_list_hybrid)/ len(record_list_hybrid))
    print(sum(extract_list_hybrid)/ len(extract_list_hybrid))
    print(sum(classifier_list_hybrid)/ len(classifier_list_hybrid))
    print(sum(record_list_server)/ len(record_list_server))
    print(sum(classifier_list_server)/ len(classifier_list_server))
    print(sum(record_list_local)/ len(record_list_local))
    print(sum(classifier_list_local)/ len(classifier_list_local))

    power_list_server = get_power_list(time_list_server, 'cloudcomputing_100_no_toggle.csv')  
    power_list_hybrid = get_power_list(time_list_hybrid, 'hybridcomputing_100_model.csv')  
    power_list_local = get_power_list(time_list_local, 'devicecomputing_100_edge.csv')
    
    power_list_server.remove(power_list_server[0])
    power_list_hybrid.remove(power_list_hybrid[0])
    power_list_local.remove(power_list_local[0])
    
    power_record_list_server, [], power_classifier_list_server = organize_power_list(power_list_server, 1)
    power_record_list_hybrid, power_extract_list_hybrid, power_classifier_list_hybrid = organize_power_list(power_list_hybrid, 0)
    power_record_list_local, [], power_classifier_list_local = organize_power_list(power_list_local, 1)

    add_results(power_record_list_local, power_classifier_list_local, power_record_list_server, power_classifier_list_server, power_record_list_hybrid, power_extract_list_hybrid, power_classifier_list_hybrid)

    pyplot.scatter(range(1,21), power_record_list_local, 20, alpha=0.5, label='Recording (Local)')
    pyplot.scatter(range(1,21), power_classifier_list_local, 20, alpha=0.5, label='Classifying (Local)')
    pyplot.scatter(range(1,21), power_record_list_hybrid, 20, alpha=0.5, label='Recording (Hybrid)')
    pyplot.scatter(range(1,21), power_extract_list_hybrid, 20, alpha=0.5, label='Extracting (Hybrid)')
    pyplot.scatter(range(1,21), power_classifier_list_hybrid, 20, alpha=0.5, label='Classifying (Hybrid)')
    pyplot.scatter(range(1,21), power_record_list_server, 20, alpha=0.5, label='Recording (Server)')
    pyplot.scatter(range(1,21), power_classifier_list_server, 20, alpha=0.5, label='Classifying (Server)')
    pyplot.legend(loc='best')
    pyplot.ylabel('Average Power (mW)')
    pyplot.xlabel('Program Iterations')
    pyplot.xticks(list(range(0,21))[::5])
    pyplot.savefig('power_test1.png')
    pyplot.close()

    #print(power_record_list_server)
    #print(power_record_list_local)

    print(sum(power_record_list_server)/ len(power_record_list_server))
    print(sum(power_classifier_list_server)/ len(power_classifier_list_server))
    print(sum(power_record_list_hybrid)/ len(power_record_list_hybrid))
    print(sum(power_extract_list_hybrid)/ len(power_extract_list_hybrid))
    print(sum(power_record_list_local)/ len(power_record_list_local))
    print(sum(power_classifier_list_local)/ len(power_classifier_list_local))
    
