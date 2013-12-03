'''
Created on 11 oct. 2013

@author: Gen
'''
import Orange

# Function returning attributes information of a dataset
def show_values(data, heading):
    print heading
    for a in data.domain.attributes:
        print "%s: %s" % (a.name, \
          reduce(lambda x,y: x+', '+y, [i for i in a.values]))
        
# Data import        
data = Orange.data.Table("..\glass.arff")

# Discretization
data_ent = Orange.data.discretization.DiscretizeTable(data, method=Orange.feature.discretization.Entropy())

# Manipulation of the discretized data
for attr in data_ent.domain.attributes :
    #Reset renamed attributes name to original ones
    if (attr.name[0:2] == "D_"):
        attr.name = attr.name[2:]
    #Replace ',' occurring in interval-valued data instances by ';' 
    attr.values = [val.replace(',',";") for val in attr.values]


show_values(data_ent, "Entropy based discretization")
print

# save the discretized data
data_ent.save('glass_dis.arff')
