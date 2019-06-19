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
    time_list = []

    time_counter = 0
    time_list.append(0)
    for key, value in time_dict.items():
        time_dict[key] = [float(value['End']) - float(value['Middle']), float(value['Middle']) - float(value['Start'])]
        record_list.append(time_dict[key][1])
        classifier_list.append(time_dict[key][0])
        time_counter += time_dict[key][1]
        time_list.append(time_counter)
        time_counter += time_dict[key][0]
        time_list.append(time_counter)
    return record_list, classifier_list, time_list

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
            if key < higher_bound and key > lower_bound:
                ave_value += value[1]
                counter += 1
        
        if counter != 0:
            ave_value = ave_value/counter
        power_list.append([higher_bound, lower_bound, ave_value])
    
    return power_list

def organize_power_list(power_list):
    power_record_list = []
    power_classifier_list = []
    switch = 0
    for power_value in power_list:
        if switch == 0:
            power_record_list.append(power_value[2])
            switch = 1
        else:
            power_classifier_list.append(power_value[2])
            switch = 0
    
    return power_record_list, power_classifier_list

if __name__ == "__main__":
    data_file_local = open("localtime.txt", "r")
    data_file_edge = open("edgetime.txt", "r")

    times_local = get_times(data_file_local)
    times_edge = get_times(data_file_edge)

    record_list_local, classifier_list_local, time_list_local= organize_data(times_local)
    record_list_edge, classifier_list_edge, time_list_edge = organize_data(times_edge)

    #print(time_list_local)
    #print(time_list_edge)
    
    pyplot.scatter(range(1,21), record_list_local, 20, alpha=0.5, label='Recording (Local)')
    pyplot.scatter(range(1,21), classifier_list_local, 20, alpha=0.5, label='Classifying (Local)')
    pyplot.scatter(range(1,21), record_list_edge, 20, alpha=0.5, label='Recording (Edge)')
    pyplot.scatter(range(1,21), classifier_list_edge, 20, alpha=0.5, label='Classifying (Edge)')
    pyplot.legend(loc='upper right')
    pyplot.ylabel('Time (s)')
    pyplot.xlabel('Program Iterations')
    pyplot.xticks(list(range(0,21))[::5])
    pyplot.savefig('runtime.png')
    pyplot.close()

    print(sum(record_list_edge)/ len(record_list_edge))
    print(sum(classifier_list_edge)/ len(classifier_list_edge))
    print(sum(record_list_local)/ len(record_list_local))
    print(sum(classifier_list_local)/ len(classifier_list_local))

    power_list_local = get_power_list(time_list_local, 'devicecomputing2.csv')
    power_list_edge = get_power_list(time_list_edge, 'cloudcomputing3.csv')    
   
    power_list_edge.remove(power_list_edge[0])
    power_list_local.remove(power_list_local[0])

    power_record_list_edge, power_classifier_list_edge = organize_power_list(power_list_edge)
    power_record_list_local, power_classifier_list_local = organize_power_list(power_list_local)
       
    pyplot.scatter(range(1,21), power_record_list_local, 20, alpha=0.5, label='Recording (Local)')
    pyplot.scatter(range(1,21), power_classifier_list_local, 20, alpha=0.5, label='Classifying (Local)')
    pyplot.scatter(range(1,21), power_record_list_edge, 20, alpha=0.5, label='Recording (Edge)')
    pyplot.scatter(range(1,21), power_classifier_list_edge, 20, alpha=0.5, label='Classifying (Edge)')
    pyplot.legend(loc='best')
    pyplot.ylabel('Average Power (mW)')
    pyplot.xlabel('Program Iterations')
    pyplot.xticks(list(range(0,21))[::5])
    pyplot.savefig('power.png')

    print(sum(power_record_list_edge)/ len(power_record_list_edge))
    print(sum(power_classifier_list_edge)/ len(power_classifier_list_edge))
    print(sum(power_record_list_local)/ len(power_record_list_local))
    print(sum(power_classifier_list_local)/ len(power_classifier_list_local))
    
