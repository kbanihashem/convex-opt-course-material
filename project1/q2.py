from logic import *
from data_generator import * 

np.set_printoptions(precision=6, suppress=True)

def eigth_twelve(should_plot=False, num_tests=1, random_state=0):
    m = 100
    n = 500
    for i in range(num_tests):
        print(f'{i + 1}th data point')
        data = get_data(m, n, random_state=i + random_state, part='a')
        A, b, c, x0 = data
        x, w, log = infeasible_centering_step(*data)
        f_log = log['f_value']
        r_primal_log = log['r_primal_norm']
        r_dual_log = log['r_dual_norm']
        r_log = log['r_norm']
        num_steps = log['num_steps']
        print(f'num_steps: {num_steps}')

        #evaluating kkt
        primal_slackness = np.max(np.abs(A @ x - b))
        dual_slackness = np.max(np.abs(grad(x, c) + A.T @ w))
        print(f'maxed out iterations: {log["maxed_out_iterations_"]}')
        print(f'primal_slackness: {primal_slackness:.6f}')
        print(f'dual_slackness: {dual_slackness:.6f}')

        cvx_output = solve_with_cvx(A, b, c, with_log=True)
        print(f'Our optimal value: {f(x, c):.6f}')
        print(f'CVX optimal value: {cvx_output["obj_value"]:.6f}')
        cvx_x = cvx_output['x_value']
        relative_error = l2(cvx_x - x) / l2(cvx_x)
        print(f'Relative error: {relative_error:.6f}')

        #plotting if necessary
        if should_plot:
            logs = {
                    'f value': f_log[:-1] - f_log[-1],
                    'r norm': r_log,
                    'r primal norm', r_primal_log,
                    'r dual norm', r_dual_log,
                    }
            for name, log in [r_primal_log, r_dual_log, r_log, f_log[:-1] - f_log[-1]]:
                num_iter = len(log)
                plt.plot(np.arange(num_iter), np.log(log), marker = 'o', linestyle='--')
                plt.title(f'log of {name}')
                plt.xlabel(iteration)
                plt.ylabel(f'log({name})')
                plt.show()
    #observing the effects of alpha, beta
    if should_plot:
        data = get_data(m, n, random_state=random_state, part='a')
        A, b, c, x0 = data
        number_of_alphas = 10
        number_of_betas = 10
        to_plot = np.zeros((number_of_alphas, number_of_betas))
        possible_alphas = np.linspace(0.1, 0.4, number_of_alphas)
        possible_betas = np.linspace(0.1, 0.9, number_of_betas)
        for i, alpha in enumerate(possible_alphas):
            for j, beta in enumerate(possible_betas):
                x, w, log = infeasible_centering_step(*data, alpha=alpha, beta=beta)
                num_steps = log['num_steps']
                to_plot[i, j] = num_steps
        plt.imshow(to_plot, cmap='hot', interpolation='nearest', extent=[0.1, 0.4, 0.1, 0.9])
        plt.title('alpha,beta heatmap')
        plt.xlabel('alpha')
        plt.ylabel('beta')
        plt.colorbar()
        plt.show()

