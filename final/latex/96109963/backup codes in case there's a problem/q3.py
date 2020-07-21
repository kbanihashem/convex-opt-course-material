import json
import numpy as np
import cvxpy as cp
print('loading data')
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
print('Data loaded')

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

#returns indexes of all reactions in subsystem 'name'
def get_subystem_indexes(name):
    return get_knockout_index({name})

def part_a():
    #should run read_data before this
    p_a, v_a, solver_a = solve_with_removed_index()
    return p_a, v_a, solver_a

def evaluated_diff(v_w, v_tilde):
    return (v_w[-1] - v_tilde[-1]) / v_w[-1]

def part_b():
    p_transport, v_transport, solver_transport = solve_with_removed_index(get_subystem_indexes(knockout_names[0]))
    p_fatty, v_fatty, solver_fatty = solve_with_removed_index(get_subystem_indexes(knockout_names[1]))
    return p_transport, v_transport, solver_transport, p_fatty, v_fatty, solver_fatty

def part_c(v_a, verbose=True):
    possible_indexes = get_subystem_indexes(knockout_names[0])
    if verbose:
        print('length of list: ', len(possible_indexes))
    li = []
    for i, index in enumerate(possible_indexes):
        if verbose:
            print(f'starting {i}th index {index}, name={name[index]}')
        p_index, v_index, solver_index = solve_with_removed_index([index])
        evaluation = evaluated_diff(v_a, v_index)
        made_it = evaluation > 0.2
        if verbose:
            print(f'status: {p_index.status}, solver: {solver_index}, evaluation: {evaluation}, made it: {made_it}')
        if made_it:
            li.append(name[index])
    return li

def main():
    p_a, v_a, solver_a = part_a()
    print('part a: ')
    print('a: status: ', p_a.status, 'value (optimal rate): ', p_a.value, 'solver: ', solver_a)
    print('part b: ')
    p_transport, v_transport, solver_transport, p_fatty, v_fatty, solver_fatty = part_b()
    print('transport: status: ', p_transport.status, 'value: ', p_transport.value, 'solver: ', solver_transport)
    print('fatty: status: ', p_fatty.status, 'value: ', p_fatty.value, 'solver: ', solver_fatty)
    print('transport and fatty diffs: ')
    print(evaluated_diff(v_a, v_transport), evaluated_diff(v_a, v_fatty))
    li_c = part_c(v_a, verbose=False)
    print('part c: ')
    print(li_c)

main()
