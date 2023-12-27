from matplotlib import pyplot
import numpy as np

x_axis = np.arange(4, 13, 4)

local_lat = [1.059328882423782, 1.059328882423782, 5.653602998769096]
server_lat = [9.830215831810248, 13.171506098243182, 305.11674312782065]
width = 1

pyplot.bar(x_axis - width/2, local_lat, width=width,label="Classify Locally", align="center", color="blue")
pyplot.bar(x_axis + width/2, server_lat, width=width,label="Classify in Server", align="center", color="orange")

pyplot.text(3.15, 1.06 + .75, str(1.06), color='black', fontweight='bold', ha='left', va='baseline')
pyplot.text(4.15, 9.83 + .75, str(9.83), color='black', fontweight='bold', ha='left', va='baseline')

pyplot.text(7.15, 1.06 + .75, str(1.06), color='black', fontweight='bold', ha='left', va='baseline')
pyplot.text(8, 13.17 + .75, str(13.17), color='black', fontweight='bold', ha='left', va='baseline')

pyplot.text(11.15, 5.65 + .75, str(5.65), color='black', fontweight='bold', ha='left', va='baseline')
pyplot.text(11.875, 305.12 + .75, str(305.12), color='black', fontweight='bold', ha='left', va='baseline')

pyplot.legend(loc='best')
pyplot.grid(which='major', linestyle = '-', linewidth = 0.5)
pyplot.minorticks_on()
pyplot.grid(which='minor', linestyle='--', linewidth = 0.5)
pyplot.xticks([4,8,12])
pyplot.ylabel('Latency (ms)')
pyplot.xlabel('Number of Pis')
pyplot.savefig('latency_noise.png')
pyplot.close()