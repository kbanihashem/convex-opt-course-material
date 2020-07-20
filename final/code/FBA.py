import json
import numpy as np
import cvxpy as cp
def get_data():
with open("Recon3D.json", "r") as file_handle:
    dictionary = json.load(file_handle);
    data = dictionary["reactions"];
    temp_name = [(data[i])["name"] for i in range(len(data))];
    b = [i for i in range(len(data)) if temp_name[i] == "Generic Human Biomass Reaction"];
    b = b[0];
    order = [i for j in (range(b), range(b+1, len(data)), range(b,b+1)) for i in j];
    name = [(data[i])["name"] for i in order];
    lower_bound = [(data[i])["lower_bound"] for i in order];
    upper_bound = [(data[i])["upper_bound"] for i in order];
    subsystem = [(data[i])["subsystem"] for i in order];
    metabolites = [(data[i])["metabolites"] for i in order];
    id = [((dictionary["metabolites"])[i])["id"] for i in range(len(dictionary["metabolites"]))];
    S = np.zeros((len(id), len(metabolites)));
    for i in range(len(metabolites)):
        for j in range(len(id)):
            if id[j] in metabolites[i].keys():
                S[j, i] = metabolites[i][id[j]];

def solve_with_removed_index(removed=None, solver=None):
    solver = solver if solver is not None else cp.ECOS
    u = np.array(upper_bound.copy())
    l = np.array(lower_bound.copy())
    if removed is not None:
        removed = np.array(removed)
        u[removed] = 0
        l[removed] = 0
    m, n = S.shape
    v = cp.Variable(n)
    #part a
    constraints = [
            S @ v == 0,
            v <= u,
            v >= l,
            ]
    obj = cp.Maximize(v[-1])
    problem = cp.Problem(obj, constraints)
    ans = problem.solve(solver=solver)
    return problem, v.value

def get_knockout_index(knockout):
    return [i for i, sub_name in enumerate(subsystem) if sub_name in knockout]

def get_subystem_indexes(name):
    return get_knockout_index({name})

knockout = set([
        'Transport, nuclear',
        'Fatty acid oxidation',
        ])

