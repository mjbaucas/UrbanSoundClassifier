# UrbanSoundClassifier
The UrbanSoundClassifier provides a test environment for urban sound classification using the UrbanSound8K dataset from https://urbansounddataset.weebly.com/. The environment makes use of Client-Server socket communication, written in Python3.6. The clients were Raspberry Pis connected to a digital MEMS microphone for sound collection. 

The environment comes with three configurations each focusinig on a certain aspect of the network (i.e. Edge, Cloud, Hybrid)

# What to run
For Edge:
  - Run edgecomp.py on the client
  - Run cloudcompsock.py on server  

For Cloud:
  - Run cloudcomp.py on client
  - Run cloudcompsock.py on server
  
For Hybrid:
  - Run hybridcomp.py on client
  - Run hybridcompsock.py on server

# Related publication
Please cite this paper if you use any code from this test environment in a publication.

M. Baucas and P. Spachos, "Using cloud and fog computing for large scale IoT-based urban sound classification," in Elsevier SIMPAT.
doi: 10.1016/j.simpat.2019.102013
