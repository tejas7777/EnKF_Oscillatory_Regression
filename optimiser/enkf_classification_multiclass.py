import torch
import copy

class EnKFOptimizerMultiClassification:
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
    


    def step(self, F, obs):
        for iteration in range(self.max_iterations):
            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Started")

            # Step [1] Draw K Particles
            self.Omega = torch.randn((self.theta.numel(), self.k)) * self.sigma  # Draw particles
            particles = self.theta.unsqueeze(1) + self.Omega  # Add the noise to the current parameter estimate

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Drawing {self.k} Particles completed")

            # Step [2] Evaluate the forward model using theta mean
            current_params_unflattened = self.unflatten_parameters(self.theta)
            F_current = F(current_params_unflattened)
            m, c = F_current.shape  # Batch size and number of classes
            Q = torch.zeros(m, c, self.k)  # [batch_size, num_classes, k]

            for i in range(self.k):
                perturbed_params = particles[:, i]
                perturbed_params_unflattened = self.unflatten_parameters(perturbed_params)
                F_perturbed = F(perturbed_params_unflattened)
                Q[:, :, i] = F_perturbed - F_current

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : forward model evaluation complete")

            # Step [3] Construct the Hessian Matrix Hj = Qj(transpose) x Qj + Γ
            Q_vec = Q.view(m * c, self.k)  # Reshape Q to [mc, k]
            H_j = Q_vec.T @ Q_vec + self.gamma * torch.eye(self.k)
            H_inv = torch.inverse(H_j)

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Hj and Hj inverse completed")

            # Step [4] Calculate the Gradient of loss function with respect to the current parameters
            gradient = self.misfit_gradient(F, self.theta, obs)
            gradient = gradient.view(-1, 1)  # Ensure it's a column vector

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : gradient calculation completed")

            # Step [5] Update the parameters
            adjustment = H_inv @ Q_vec.T  # Shape [k, mc]
            scaled_adjustment = self.Omega @ adjustment  # Shape [n, mc]
            update = scaled_adjustment @ gradient  # Shape [n, mc] x [mc, 1] = [n, 1]
            update = update.view(-1)  # Reshape to [n]

            self.theta -= self.lr * update  # Now both are [n]

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

    def misfit_gradient(self, F, theta, d_obs):
        # Unflatten parameters
        params_unflattened = self.unflatten_parameters(theta)
        
        # Forward pass
        logits = F(params_unflattened)
        #predictions = torch.sigmoid(logits)
        num_classes = logits.shape[1]
        d_obs = torch.nn.functional.one_hot(d_obs, num_classes=num_classes).float()
        
        # Compute gradient of loss with respect to predictions
        residuals = logits - d_obs 
        
        return residuals.view(-1, 1)


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







