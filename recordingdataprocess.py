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
        #if 'Middle' in value:
        #    time_dict[key] = [float(value['End']) - float(value['Middle']), float(value['Middle']) - float(value['Start'])]
        #    record_list.append(time_dict[key][1])
        #    classifier_list.append(time_dict[key][0])
        #    time_counter += time_dict[key][1]
        #    time_list.append(time_counter)
        #    time_counter += time_dict[key][0]
        #    time_list.append(time_counter)
        #else:
        time_dict[key] = [float(value['End']) - float(value['Start'])]
        record_list.append(time_dict[key][0])
        time_counter += time_dict[key][0]
        time_list.append(time_counter)
    return record_list[10:], classifier_list[10:], time_list[10:]

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

def organize_power_list(power_list, value):
    power_record_list = []
    power_send_list = []
    switch = 0
    for power_value in power_list:
        if switch == 0 or value == 1:
            power_record_list.append(power_value[2])
            switch = 1
        else:
            power_send_list.append(power_value[2])
            switch = 0
        
    return power_record_list, power_send_list

if __name__ == "__main__":
    data_file_record = open("recordtime.txt", "r")
    data_file_send = open("recordsendtime.txt", "r")

    times_record = get_times(data_file_record)
    times_send = get_times(data_file_send)
    
    record_list, [], time_list = organize_data(times_record)
    record_list_s, send_list, time_list_s = organize_data(times_send)


    power_list_record = get_power_list(time_list, 'recordingpower6.csv')
    power_list_send = get_power_list(time_list_s, 'recordingpowersend4.csv') 

    power_list_record.remove(power_list_record[0])
    power_list_send.remove(power_list_send[0])
    
    
    power_record_list, [] = organize_power_list(power_list_record, 1)
    power_send_list, []= organize_power_list(power_list_send, 1)

    pyplot.scatter(range(1,21), power_record_list, 20, alpha=0.5, label='Record Only')
    pyplot.scatter(range(1,21), power_send_list, 20, alpha=0.5, label='Record and Send')
    pyplot.legend(loc='best')
    pyplot.ylabel('Average Power (mW)')
    pyplot.xlabel('Program Iterations')
    pyplot.xticks(list(range(0,21))[::5])
    pyplot.savefig('power_record.png')

    diff = []
    for x in range(0, 20):
        diff.append(power_send_list[x] - power_record_list[x])
    
    print(sum(diff)/20)
    print(sum(power_send_list)/20)
    print(sum(power_record_list)/20)
