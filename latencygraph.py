from matplotlib import pyplot

pyplot.bar([3,7,11], [0.6, 1.0, 5.4], width=1,label="Local", align="center")
pyplot.bar([4,8,12], [9.7, 13.0, 300.7], width=1,label="Server", align="center")
#pyplot.bar([4.5,8.5,12.5], [1.7, 2.2, 5.5], width=0.5,label="Configuration C", align="center")
pyplot.legend(loc='best')
pyplot.xticks([4,8,12])
pyplot.ylabel('Latency (ms)')
pyplot.xlabel('Number of Pis')
pyplot.savefig('latency_noise.png')
pyplot.close()