from matplotlib import pyplot
from math import log10

pyplot.bar([3,7,11], [0.6, 1.0, 5.4], width=1,label="Configuration A", align="center", color="purple")
pyplot.bar([4,8,12], [9.7, 13.0, 300.7], width=1,label="Configuration B", align="center", color="red")
#pyplot.bar([5,9,13], [1.7, 2.2, 5.5], width=1,label="Proposed Configuration", align="center")

pyplot.text(2.7, 0.65, str(0.6), color='black', fontweight='bold', ha='left', va='baseline')
pyplot.text(3.7, 9.7 + .50, str(9.7), color='black', fontweight='bold', ha='left', va='baseline')
#pyplot.text(4.7, 1.8, str(1.7), color='black', fontweight='bold', ha='left', va='baseline')
pyplot.text(6.7, 1.07, str(1.0), color='black', fontweight='bold', ha='left', va='baseline')
pyplot.text(7.62, 13.0 + .50, str(13.0), color='black', fontweight='bold', ha='left', va='baseline')
#pyplot.text(8.7, 2.3, str(2.2), color='black', fontweight='bold', ha='left', va='baseline')
pyplot.text(10.71, 5.3 + .50, str(5.4), color='black', fontweight='bold', ha='left', va='baseline')
pyplot.text(11.5, 300.7 + .50, str(300.7), color='black', fontweight='bold', ha='left', va='baseline')
#pyplot.text(12.71, 5.4 + .50, str(5.5), color='black', fontweight='bold', ha='left', va='baseline')

pyplot.legend(loc='best')
pyplot.xticks([4,8,12])
pyplot.ylabel('Latency (ms)')
pyplot.xlabel('Number of Pis')
pyplot.yscale('log')
pyplot.savefig('latency_locserv.png')
pyplot.close()