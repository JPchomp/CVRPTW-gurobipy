# Import the necessary libraries
from gurobipy import *
from gurobipy import Model
from gurobipy import GRB
from gurobipy import quicksum
import os
import xlrd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

## TODO: kilometer matrix 
## TODO: take itins to 


# Open an Excel workbook named "time window.xls"
book = xlrd.open_workbook(os.path.join("time window.xls"))

# Initialize data storage variables
Node = []
Demand = {}           # Demand in Thousands
ServiceTime = {}      # Service time in minutes
Distance = {}         # Distance in kms
TravelTime = {}       # Travel time in minutes
VehicleID = []       # Vehicle number
VehicleCap = {}
dispatch_cost = {}
max_distance_per_vehicle = {}
# Cap = 20 unused
ai = {}
bi = {}
loni = {}
lati = {}
Arc = {}

# Read data from the "demand" sheet of the Excel file
sh = book.sheet_by_name("demand")
i = 1

while True:
    try:
        sp = sh.cell_value(i, 0)
        Node.append(sp)
        Demand[sp] = sh.cell_value(i, 1)
        ServiceTime[sp] = sh.cell_value(i, 2)
        ai[sp] = sh.cell_value(i, 3)
        bi[sp] = sh.cell_value(i, 4)
        loni[sp] = sh.cell_value(i,5)
        lati[sp] = sh.cell_value(i,6)
        i = i + 1
    except IndexError:
        break

# Read data from the "VehicleID" sheet
sh = book.sheet_by_name("VehicleNum")
veh_list = sh.col_values(0)

# Set the number of vehicles

# Its one less because of the header.
numberOfVehicle = len(veh_list)-1
K = numberOfVehicle

i = 1

while True:
    try:
        sp = sh.cell_value(i, 0)
        VehicleID.append(sp)

        spp = sh.cell_value(i,1)
        VehicleCap[sp] = spp

        sppp = sh.cell_value(i,2)
        dispatch_cost[sp] = sppp

        spppp = sh.cell_value(i,3)
        max_distance_per_vehicle[sp] = spppp

        i = i + 1
    except IndexError:
        break

# Initialize the cost matrix
cost = {}
sh = book.sheet_by_name("Cost")
i = 1
for P in Node:
    j = 1
    for Q in Node:
        cost[P, Q] = sh.cell_value(i, j)
        j += 1
    i += 1

# Initialize the distance matrix
sh = book.sheet_by_name("Distance")
i = 1
for P in Node:
    j = 1
    for Q in Node:
        Distance[P, Q] = sh.cell_value(i, j)
        j += 1
    i += 1

# Initialize the travel time matrix
sh = book.sheet_by_name("TravelTime")
i = 1
for P in Node:
    j = 1
    for Q in Node:
        TravelTime[P, Q] = sh.cell_value(i, j)
        j += 1
    i += 1

# Initialize the arc availability matrix
Aij = {}
sh = book.sheet_by_name("Aij")
i = 1

for P in Node:
    j = 1
    for Q in Node:
        Aij[P, Q] = sh.cell_value(i, j)
        j += 1
    i += 1

# Create a Gurobi optimization model with the name "Time window 1"
m = Model("Time window 1")

# Set the optimization objective to minimize the total cost
m.modelSense = GRB.MINIMIZE


# Define binary decision variables xijk to represent vehicle routes
xijk = m.addVars(Node, Node, VehicleID, vtype=GRB.BINARY, name='X_ijk')

# Define continuous variables Tik to represent the time at each node
Tik = m.addVars(Node, VehicleID, vtype=GRB.CONTINUOUS, name='T_ik')

# Define binary decision variables to represent whether each vehicle is dispatched
is_dispatched = m.addVars(VehicleID, vtype=GRB.BINARY, name='is_dispatched')

# Define auxiliary variables for positive differences between demand and quantity served
positive_demand_diff = m.addVars(Node, lb=0, name='positive_demand_diff')

# Set up penalty in the objective function for unfulfilled demands
penalty_coefficient = 9999999999999999

m.setObjective(
    # Existing cost components
    sum((cost[i, j] * xijk[i, j, k]  for i in Node for j in Node for k in VehicleID if Aij[i, j] == 1)) +

    sum(dispatch_cost[k] * is_dispatched[k] for k in VehicleID) +

    # Penalty for unmet demands
    penalty_coefficient * sum(positive_demand_diff[i] for i in Node if i != 'DepotStart' and i != 'DepotEnd')
)

M = 1000000


# Add constraints to enforce positive_demand_diff
for i in Node:
    if i != 'DepotStart' and i != 'DepotEnd':
        m.addConstr(
            positive_demand_diff[i] >= Demand[i] - quicksum(xijk[i, j, k] for j in Node for k in VehicleID if Aij[i, j] == 1),
            f"PositiveDemandDiff_{i}"
        )

# Add subtour elimination constraints considering the is_dispatched variable
for i in Node:
    for j in Node:
        for k in VehicleID:
            if Aij[i, j] == 1 and i != 'DepotEnd' and j != 'DepotStart' and j != 'DepotEnd':
                m.addConstr(
                    (Tik[i, k] + ServiceTime[i] + TravelTime[i, j] - Tik[j, k]) <= (1 - xijk[i, j, k]) * M ,
                    f"SubtourElimination_{i}_{j}_{k}"
                )

for i in Node:
    if i != 'DepotStart' and i != 'DepotEnd':
        m.addConstr(
            sum(xijk[i, j, k] for j in Node if j != i for k in VehicleID) == 1,
            f"SubtourElimination_{i}"
        )


# for i in Node:
#     if i != 'DepotStart' and i != 'DepotEnd':
#         for j in Node:
#             for k in VehicleID:
#                 m.addConstr(
#                     xijk[i, j, k] <= is_dispatched[k],
#                     f"DispatchCondition_{i}_{k}"
#                 )


# Add constraints to ensure that the time of arrival at each node falls within its time window considering vehicle dispatch
for i in Node:
    for k in VehicleID:
        for j in Node:
            if Aij[i, j] == 1:
                # Time window constraints with consideration for dispatched vehicles
                m.addConstr(
                    Tik[i, k] >= ai[i] * is_dispatched[k],
                    f"TimeWindow_Lower_{i}_{k}"
                )
                m.addConstr(
                    Tik[i, k] * is_dispatched[k] <= bi[i],
                    f"TimeWindow_Upper_{i}_{k}"
                )

# Add constraints to ensure that each non-depot node is visited exactly at least once considering vehicle dispatch
for i in Node:
    if i != 'DepotStart' and i != 'DepotEnd':
        for k in VehicleID:
            m.addConstr(
                sum(xijk[i, j, k] for j in Node if Aij[i, j] == 1) >= is_dispatched[k],
                f"VisitOnce_{i}_{k}"
            )

# Add constraints to ensure that the starting depot is visited exactly once by each dispatched vehicle
for k in VehicleID:
    m.addConstr(
        sum(xijk['DepotStart', j, k] for j in Node if Aij['DepotStart', j] == 1) == is_dispatched[k],
        f"VisitDepotStart_{k}"
    )

# Add constraints to ensure that the ending depot is visited exactly once by each dispatched vehicle
for k in VehicleID:
    m.addConstr(
        sum(xijk[i, 'DepotEnd', k] for i in Node if Aij[i, 'DepotEnd'] == 1) == is_dispatched[k],
        f"VisitDepotEnd_{k}"
    )


# Add constraints to maintain flow conservation in the network considering vehicle dispatch
for j in Node:
    for k in VehicleID:
        if j != 'DepotStart' and j != 'DepotEnd':
            for i in Node:
                m.addConstr(
                    sum(xijk[i, j, k] for i in Node if Aij[i, j] == 1) - sum(xijk[j, i, k] for i in Node if Aij[j, i] == 1) == 0,
                    f"FlowConservation_{j}_{k}"
                )


# Add constraints to limit the total distance traveled by each vehicle
for k in VehicleID:
    m.addConstr(
        quicksum(Distance[i, j] * xijk[i, j, k] * is_dispatched[k] for i in Node for j in Node if Aij[i, j] == 1) <= max_distance_per_vehicle[k] ,
        f"MaxDistance_{k}"
    )

# Add capacity constraints to limit the total demand carried by each vehicle
for k in VehicleID:
    m.addConstr((sum(Demand[i] * xijk[i, j, k] * Aij[i, j] * is_dispatched[k] for i in Node for j in Node)) <= VehicleCap[k])


# Solve the optimization problem
m.optimize()



###### 
# Check if the optimization was successful before accessing all variable values
if m.status == GRB.OPTIMAL:

    # Write the model to a file in LP format for inspection or debugging
    m.write('Timewindow.lp')

    # Print the variables with non-zero values (selected routes) and the objective value (total cost)
    for v in m.getVars():
        if v.x > 0.01:
            print(v.varName, v.x)
    print('Objective:', m.objVal)


    for k in VehicleID: print(is_dispatched[k])

    # Create a list to store the routes for each vehicle
    routes = [[] for _ in range(numberOfVehicle)]

    # Extract the routes from the Gurobi model for valid indices
    for k, l in zip(VehicleID, range(numberOfVehicle)):  # Iterate over the list of vehicles
        vehicle_route = []
        for i in Node:
            for j in Node:
                if xijk[i, j, k].x > 0.5:
                    vehicle_route.append(i)  # Append the node index

        # Append the final node of the node list to the route
        final_node = Node[-1]  # Assuming the last node in the list is the final node
        vehicle_route.append(final_node)

        routes[l] = vehicle_route

    # Print the total distance traveled by each vehicle
    for k in VehicleID:
        total_distance = sum(Distance[i, j] * xijk[i, j, k].x for i in Node for j in Node if Aij[i, j] == 1)
        print(f"Total distance traveled by Vehicle {k}: {total_distance} kilometers")

    ######

    # Initialize a dictionary to store total demand fulfilled at each node
    total_demand_fulfilled = {i: 0 for i in Node}

    # Extract the solution to calculate total demand fulfilled
    for k in VehicleID:
        for i in Node:
            for j in Node:
                if Aij[i, j] == 1 and xijk[i, j, k].x > 0.5:  # If node i is visited by vehicle k
                    total_demand_fulfilled[i] += Demand[i]  # Add the demand of node i to the total demand fulfilled

# I need to add how much is delivered by each truck at each step

    # Print or use total_demand_fulfilled to analyze the total demand fulfilled at each location
    for node, demand_fulfilled in total_demand_fulfilled.items():
        print(f"Node {node}: Total Demand Fulfilled = {demand_fulfilled} Required= {Demand[node]} Postive Dem diff = {positive_demand_diff[node]}")

    ######
    # Create a color map for each vehicle ID
    color_map = {v: plt.cm.jet(i / len(VehicleID)) for i, v in enumerate(VehicleID)}

    # Create a scatter plot of latitude and longitude coordinates
    plt.figure(figsize=(10, 8))

    # Scatter plot of nodes
    for k in VehicleID:
        color = color_map[k]
        plt.scatter([loni[i] for i in routes[VehicleID.index(k)]], [lati[i] for i in routes[VehicleID.index(k)]], marker='o', color=color, label=f'Nodes - Vehicle {k}')

            # Add labels for each node
        for node in routes[VehicleID.index(k)]:
            plt.text(loni[node], lati[node], f'{node}', fontsize=8, ha='right')


    # Plot the routes for each vehicle with arrows
    for k, route in enumerate(routes):
        color = color_map[VehicleID[k]]
        for i in range(len(route) - 1):
            x_start, y_start = loni[route[i]], lati[route[i]]
            x_end, y_end = loni[route[i + 1]], lati[route[i + 1]]

            arrow = patches.FancyArrowPatch((x_start, y_start), (x_end, y_end), color=color, arrowstyle='->', mutation_scale=15)
            plt.gca().add_patch(arrow)

    # Set axis labels and legend
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(loc='best')

    # Show the plot
    plt.grid(True)
    plt.show()

else:
    print("Optimization was not successful.")