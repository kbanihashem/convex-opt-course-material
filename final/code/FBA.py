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

def solve_with_knockout(knockout=None):
    u = upper_bound.copy()
    l = lower_bound.copy()
    if knockout is not None:
        for i, col_name in enumerate(name):
            if col_name in knockout:
                u[i] = 0
                l[i] = 0
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
    ans = problem.solve()
    return problem, v.value
