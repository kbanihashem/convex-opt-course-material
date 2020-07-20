import json
import numpy as np
import cvxpy as cp
def read_data():
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
    knockout_names = [
            'Transport, nuclear',
            'Fatty acid oxidation',
            ]

def solve_with_removed_index(removed=None):
    solvers = {'ECOS': cp.ECOS, 'QSOP': cp.OSQP, 'SCS': cp.SCS}
    for solver_name, solver in solvers.items():
        u = np.array(upper_bound.copy())
        l = np.array(lower_bound.copy())
        if removed is not None:
            removed = np.array(removed)
            u[removed] = 0
            l[removed] = 0
        m, n = S.shape
        v = cp.Variable(n)
        constraints = [
                S @ v == 0,
                v <= u,
                v >= l,
                ]
        obj = cp.Maximize(v[-1])
        problem = cp.Problem(obj, constraints)
        ans = problem.solve(solver=solver)
        if problem.status == 'optimal':
            return problem, v.value, solver_name

def get_knockout_index(knockout):
    return [i for i, sub_name in enumerate(subsystem) if sub_name in knockout]

def get_subystem_indexes(name):
    return get_knockout_index({name})

def part_a():
    #should run read_data before this
    global p_a, v_a, solver_a
    p_a, v_a, solver_a = solve_with_removed_index()

def evaluated_diff(v_w, v_tilde):
    return (v_w[-1] - v_tilde[-1]) / v_w[-1]

def part_b():
    global p_transport, v_transport, solver_transport
    global p_fatty, v_fatty, solver_fatty
    p_transport, v_transport, solver_transport = solve_with_removed_index(get_subystem_indexes(knockout_names[0]))
    p_fatty, v_fatty, solver_fatty = solve_with_removed_index(get_subystem_indexes(knockout_names[1]))

def main():
    read_data()
    part_a()
    part_b()
    print(p_a.status, p_transport.status, p_fatty.status)
    print(p_a.value, p_transport.value, p_fatty.value)
    print(evaluated_diff(v_a.value, v_transport.value), evaluated_diff(v_a.value, v_fatty.value))

if __name__ == '__main__':
    main()
