#%%
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import Model, GRB

#%% INPUT
#Create a graph

G = nx.DiGraph()
n_vehicles = 5 #number of vehicles
n_customers = 50 #number of customers
capacity = 5 #capacity of vehicles
O = 0 #origin node
S = n_customers+1 #destination node
speed = 1 #speed of vehicles
c_travel_time = 1 #cost of travel time
c_vehicle = 10 #cost of vehicle

V = [i for i in range(1, n_vehicles+1)] #set of vehicles
N = [i for i in range(1, n_customers+1)] #set of customers

#add nodes to graph
G.add_node(O, demand = 0, x = 0, y = 0, upper_TW = 10000, lower_TW = 0)
for i in N:
    G.add_node(i+1, demand = 1, x = 10*np.random.random(), y = 10*np.random.random(), upper_TW = 10000, lower_TW = 0) #upper and lower time windows
G.add_node(S, demand = 0, x = 0, y = 0, upper_TW = 10000, lower_TW = 0)

# add edges to graph
for i in G.nodes:
    for j in G.nodes:
        if i!=j:
            G.add_edge(i,j, travel_time = np.sqrt((G.nodes[i]['x']-G.nodes[j]['x'])**2 + (G.nodes[i]['y']-G.nodes[j]['y'])**2)/speed)



#%% CREATE MODEL AND VARIABLES

mdl = Model("CVRPTW")


#routing variable: 1 if vehicle v travels from node i to node j, 0 otherwise
x = { 
    (v, i, j): mdl.addVar(name="x_{0}_{1}_{2}".format(v, i, j), vtype=GRB.BINARY)
    for v in V for i, j in G.edges
}
#time variable: time at which vehicle v arrives at node i
w = {
    (v, i): mdl.addVar(name="w_{0}_{1}".format(v, i), vtype=GRB.CONTINUOUS, lb=0)
    for v in V for i in G.nodes
}

#%% OBJECTIVE AND CONSTRAINTS

# Objective
mdl.setObjective(
    #min total distance or travel time
    c_travel_time*sum(
        x[v, i, j] * G.edges[i, j]["travel_time"]
        for v, i, j in x.keys()
    )
    #min number of vehicles
    + c_vehicle*sum(
        x[v, O, j]
        for v in V for i, j in G.out_edges(O) if j != S
                        ),
    GRB.MINIMIZE
)


# customers receive only one unit of flow
mdl.addConstrs(
            sum(x[v, i, j] for v in V for i, j1 in G.in_edges(j))
            == 1
              for j in G.nodes if j != O and j != S
        )

# # conservation of flow
mdl.addConstrs(
            sum(x[v, i, j] for i1, j in G.out_edges(i))
            - sum(x[v, j, i] for j, i1 in G.in_edges(i))
            == 0 
            for v in V for i in G.nodes if i != O and i != S
        )

# vehicle departure and arrival to depots
mdl.addConstrs(
            sum(x[v, O, j] for i, j in G.out_edges(O)) == 1
            for v in V
)
mdl.addConstrs(
            sum(x[v, i, S] for i, j in G.in_edges(S)) == 1
            for v in V
)   

#each vehicle visits nodes at most once 
mdl.addConstrs(
            sum(x[v, i, j] for i, j1 in G.in_edges(j))
            <= 1 
            for v in V for j in G.nodes if j != O and j != S
        )

# time constraints
mdl.addConstrs(
            w[v, j]
            >= w[v, i]
            + G.edges[i, j]["travel_time"]
            - 10000 * (1 - x[v, i, j]) #big M
            for v, i, j in x.keys()
        )

# time windows,  only enforce when visiting
mdl.addConstrs(
            w[v, i]
            <= G.nodes[i]["upper_TW"]*sum(x[v, i1, j] for i1, j in G.out_edges(i))
            for v in V for i in G.nodes if i != O and i != S
        )
mdl.addConstrs(
            w[v, i]
            >= G.nodes[i]["lower_TW"]*sum(x[v, i1, j] for i1, j in G.out_edges(i))
            for v in V for i in G.nodes if i != O and i != S
        )

# capacity constraints
mdl.addConstrs(
            sum(x[v, i, j] * G.nodes[j]["demand"] for i, j in G.out_edges(O))
            <= capacity 
            for v in V 
        )


# solving model
#log output
mdl.setParam(GRB.Param.OutputFlag, 1)
#time limit in seconds
mdl.setParam(GRB.Param.TimeLimit, 60)
#gap stopping criteria (0 = optimal)
mdl.setParam(GRB.Param.MIPGap, 0)
mdl.optimize()
#mdl status (2: optimal, 3: infeasible, 9: time limit reached)
print("model status:", mdl.status)

sol = {}
for v in mdl.getVars():
    sol[v.varName] = v.x
sol["obj"] = mdl.objVal
sol["gap"] = mdl.MIPGap
#%%
# plot solution
plt.figure(figsize=(10, 10))

#plot depot
plt.scatter(
    [G.nodes[O]["x"]],
    [G.nodes[O]["y"]],
    color="green",
    s=100,
)

#plot nodes
plt.scatter(
    [G.nodes[i]["x"] for i in G.nodes if i != O and i != S],
    [G.nodes[i]["y"] for i in G.nodes if i != O and i != S],
    color="black",
    s=50,
)
#plot route
for v,i,j in x.keys():
    if x[v,i,j].x >0:
        if i == 0:
            #identify the first leg to know the direction
            plt.plot(
                [G.nodes[i]["x"], G.nodes[j]["x"]],
                [G.nodes[i]["y"], G.nodes[j]["y"]],
                color="red",
                alpha=0.5,
            )
        else:
            plt.plot(
                [G.nodes[i]["x"], G.nodes[j]["x"]],
                [G.nodes[i]["y"], G.nodes[j]["y"]],
                color="blue",
                alpha=0.5,
            )

plt.title("LA CONCHA DE TU MADRE ALL BOYS")

# %%