import torch
import copy

class EnKFOptimizerGradFreeMemory:
    def __init__(self, model, lr=1e-3, sigma=0.1, k=10, gamma=1e-3, max_iterations=10, debug_mode=False):
        self.model = model
        self.lr = lr
        self.sigma = sigma
        self.k = k
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.parameters = list(model.parameters())
        self.theta = torch.cat([p.data.view(-1) for p in self.parameters])  #Flattened parameters
        self.shapes = [p.shape for p in self.parameters]  #For keeping track of original shapes
        self.cumulative_sizes = [0] + list(torch.cumsum(torch.tensor([p.numel() for p in self.parameters]), dim=0))
        self.debug_mode = debug_mode
        self.Omega = None #Omega memory
        self.Q = None #Q memory


    def flatten_parameters(self, parameters):
        '''
        The weights from all the layers will be considered as a single vector
        '''
        return torch.cat([p.data.view(-1) for p in parameters])

    def unflatten_parameters(self, flat_params):
        '''
        Here, we regain the shape to so that we can use them to evaluate the model
        '''
        params_list = []
        start = 0
        for shape in self.shapes:
            num_elements = torch.prod(torch.tensor(shape))
            params_list.append(flat_params[start:start + num_elements].view(shape))
            start += num_elements
        return params_list
    


    def step(self, F, obs):
        for iteration in range(self.max_iterations):

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Started")

            '''
            Step [1] We will Draw K Particles
            '''
            Omega_j = torch.randn((self.theta.numel(), self.k)) * self.sigma  # Draw particles
            particles = self.theta.unsqueeze(1) + Omega_j  # Add the noise to the current parameter estimate

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Drawing {self.k} Particles completed")

            '''
            Step [2] Now we Evaluate the forward model using theta mean
            '''
            current_params_unflattened = self.unflatten_parameters(self.theta)
            with torch.no_grad():
                F_current = F(current_params_unflattened)
                
            Q_j = torch.zeros(obs.shape[0], self.k)  # [batch_size, k]

            for i in range(self.k):
                perturbed_params = particles[:, i]
                perturbed_params_unflattened = self.unflatten_parameters(perturbed_params)

                # Evaluate the forward model on the perturbed parameters
                with torch.no_grad():
                    F_perturbed = F(perturbed_params_unflattened)

                # Compute the difference
                Q_j[:, i] = F_perturbed.squeeze() - F_current.squeeze()

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : forward model evaluation complete")

            '''
            Step [4] Update the memory matrices
            '''

            if self.Omega is None:
                self.Omega = Omega_j
            else:
                #self.Omega = self.Omega + Omega_j
                self.Omega = torch.cat((self.Omega, Omega_j), dim = 1)

            if self.Q is None:
                self.Q = Q_j
            else:
                #self.Q = self.Q + Q_j
                self.Q = torch.cat((self.Q, Q_j), dim = 1)


            '''
            Step [3] Now we can construct the Hessian Matrix  Hj = Qj(transpose) x Qj + Γ
            '''
            # H_j = self.Q.T @ self.Q + self.gamma * torch.eye(self.k)
            gamma_matrix = self.gamma * torch.eye(self.Q.size(1))
            H_j = self.Q.T @ self.Q + gamma_matrix
            H_inv = torch.inverse(H_j)

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Hj and Hj inverse completed")

            '''
            Step [4] Calculate the Gradient of loss function with respect to the current parameters
            '''
            gradient = self.misfit_gradient(F, self.theta, obs)
            gradient = gradient.view(-1, 1)  # Ensure it's a column vector

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : gradient calculation completed")

            '''
            Step [5] Update the parameters
            '''

            # print(f"Theta shape: {self.theta.shape}")
            # print(f"Omega shape: {self.Omega.shape}")
            # print(f"H_inv shape: {H_inv.shape}")
            # print(f"Q shape: {Q.shape}")
            # print(f"Gradient shape: {gradient.shape}")

            adjustment = H_inv @ self.Q.T  # Shape [k, m]
            scaled_adjustment = self.Omega @ adjustment  # Shape [n, m]
            update = scaled_adjustment @ gradient  # Shape [n, m] x [m, 1] = [n, 1]
            update = update.view(-1)  # Reshape to [n]

            # Perform line search to determine optimal learning rate
            # lr = self.simple_line_search(F, D, gradient, scaled_adjustment, self.lr)

            lr = self.simple_line_search(F,update=update,initial_lr=self.lr,obs=obs)

            # lr = self.armijo_line_search(F,update=update,initial_lr=self.lr,obs=obs)

            #print(f"Line Search selected {lr}")

            self.theta -= lr * update  # Now both are [n]

            # Update the actual model parameters
            self.update_model_parameters(self.theta)

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : parameter update completed")
               

            


    def update_model_parameters(self, flat_params):
        idx = 0
        for param in self.model.parameters():
            #param.grad = None
            num_elements = param.numel()
            param.data.copy_(flat_params[idx:idx + num_elements].reshape(param.shape))
            idx += num_elements

    def misfit_gradient(self, F, thetha, d_obs):
        #Forward pass to get model outputs
        t = F(self.unflatten_parameters(thetha))
        
        #compute simple residuals
        residuals = t - d_obs

        return residuals.view(-1, 1)


    def simple_line_search(self, F, update, initial_lr, obs, reduction_factor=0.5, max_reductions=5):
        lr = initial_lr
        current_params_unflattened = self.unflatten_parameters(self.theta)
        
        # Compute the initial predictions and loss directly
        current_predictions = F(current_params_unflattened)
        current_loss = torch.mean((current_predictions - obs) ** 2).item()  # Compute MSE and convert to scalar

        for _ in range(max_reductions):
            new_theta = self.theta - lr * update
            new_predictions = F(self.unflatten_parameters(new_theta))
            new_loss = torch.mean((new_predictions - obs) ** 2).item()  # Compute MSE and convert to scalar

            if new_loss < current_loss:
                return lr
            
            lr -= lr*reduction_factor

        return lr






