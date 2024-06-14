import torch
import copy

class EnKFOptimizer:
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
    


    def step(self, F, D):
        for iteration in range(self.max_iterations):

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Started")
            

            '''
            Step [1] We will Draw K Particles
            '''
            self.Omega = torch.randn((self.theta.numel(), self.k)) * self.sigma  #Draw particles
            particles = self.theta.unsqueeze(1) + self.Omega  #Add the noise to the current parameter estimate

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Drawing {self.k} Particles completed")

            '''
            Step [2] Now we Evaluate the forward model
            '''
            current_params_unflattened = self.unflatten_parameters(self.theta)
            F_current = F(current_params_unflattened)
            Q = torch.zeros(1, self.k)

            for i in range(self.k):
                perturbed_params = particles[:, i]
                perturbed_params_unflattened = self.unflatten_parameters(perturbed_params)

                #Evaluate the forward model on the perturbed parameters
                F_perturbed = F(perturbed_params_unflattened)

                #Compute the difference
                Q[0, i] = (F_perturbed - F_current).mean().item()  #Store mean difference for scalar output

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : forward model evaluation complete")

            '''
            Step [3] Now we can construct the Hessian Matrix  Hj = Qj(transpose) x Qj + Γ
            '''
            H_j = Q.T @ Q + self.gamma * torch.eye(self.k)
            H_inv = torch.inverse(H_j)

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Hj and Hj inverse completed")

            '''
            Step [4] Calculate the Gradient of loss function with respect to the current parameters
            '''
            gradient = self.calculate_gradient(F, D)
            gradient = gradient.view(-1, 1)  #Ensure it's a column vector

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : gradient calculation completed")
            
            '''
            Step [5] Update the paramters
            '''

            adjustment = H_inv @ Q.T  #Shape [k, m]
            scaled_adjustment = self.Omega @ adjustment  #Shape [n, m]
            update = scaled_adjustment * gradient
            update = update.view(-1)  #Reshape to [n]

            #Perform line search to determine optimal learning rate
            lr = self.simple_line_search(F, D, gradient, scaled_adjustment, self.lr)

            self.theta -= lr * update  #Now both are [n]

            #Update the actual model parameters
            self.update_model_parameters(self.theta)

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : parameter update completed")

            # print( f"Sigma={self.sigma}, "
            # f"Gradient Norm={gradient.norm().item()}, "
            # f"Parameter Update Norm={update.norm().item()}, "
            # f"Mean Q Value={Q.mean().item()}, "
            # f"H Matrix Condition Number={torch.linalg.cond(H_j).item()}")

            


    def update_model_parameters(self, flat_params):
        idx = 0
        for param in self.model.parameters():
            #param.grad = None
            num_elements = param.numel()
            param.data.copy_(flat_params[idx:idx + num_elements].reshape(param.shape))
            idx += num_elements



    def calculate_gradient(self, F, loss, epsilon=1e-5):
        #Initialising here to an empty vector of 0s, dimensions will be similar to thetha
        grad = torch.zeros_like(self.theta) 

        for i in range(len(self.theta)):
            original_value = self.theta[i].item()

            #add positively
            self.theta[i] = original_value + epsilon
            loss_plus = loss(F(self.unflatten_parameters(self.theta)))

            #add negatively
            self.theta[i] = original_value - epsilon
            loss_minus = loss(F(self.unflatten_parameters(self.theta)))

            #Approximate derivative
            grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

            #Restore original parameter value
            self.theta[i] = original_value

        return grad
    
    def gradient_misfit_function(self, F):
        pass
    
    def calculate_jacobian_and_gradient(self, F, D):
        torch.set_grad_enabled(True)

        current_params_unflattened = self.unflatten_parameters(self.theta)

        def compute_loss(params):
            output = F(params)  # Make sure F can handle the parameter structure
            return D(output)
        
        params_tensor = torch.cat([p.detach().requires_grad_(True) for p in current_params_unflattened])
        jacobian_result = torch.autograd.functional.jacobian(compute_loss, (params_tensor,))

        if isinstance(jacobian_result, tuple):
            jacobian = jacobian_result[0]  #Assuming the first element is the relevant one
        else:
            jacobian = jacobian_result

        #Reshape jacobian to ensure it is [n x 1]
        gradient = jacobian.view(-1, 1)  # Reshape to [n x 1] explicitly

        return gradient

    def simple_line_search(self, F, D, gradient, adjustment, initial_lr, reduction_factor=0.5, max_reductions=5):

        lr = initial_lr
        current_loss = D(F(self.unflatten_parameters(self.theta)))

        for _ in range(max_reductions):
            new_theta = self.theta - lr * adjustment * gradient
            new_loss = D(F(self.unflatten_parameters(new_theta)))

            if new_loss < current_loss:
                return lr
            
            lr *= reduction_factor

        return lr






