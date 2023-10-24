import csdl
import torch


class ClModel(csdl.CustomExplicitOperation):
    def initialize(self):
        # The neural net will be a pre-trained model
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('neural_net')
        self.parameters.declare('prefix', types=str, default=None, allow_none=True)


    def define(self):
        num_nodes = self.parameters['num_nodes']
        prefix = self.parameters['prefix']

        if prefix:
            self.add_input(f'{prefix}_neural_net_input_extrap', shape=(num_nodes, 3))
            self.add_output(f'{prefix}_Cl', shape=(num_nodes, ))

            self.declare_derivatives(f'{prefix}_Cl', f'{prefix}_neural_net_input_extrap')

        else:
            self.add_input('neural_net_input_extrap', shape=(num_nodes, 3))
            self.add_output('Cl', shape=(num_nodes, ))

            self.declare_derivatives('Cl', 'neural_net_input_extrap')


    
    def compute(self, inputs, outputs):
        neural_net = self.parameters['neural_net']     
        prefix = self.parameters['prefix']

        if prefix:
            neural_net_input = torch.Tensor(inputs[f'{prefix}_neural_net_input_extrap'])
            neural_net_prediction = neural_net(neural_net_input).detach().numpy()
            outputs[f'{prefix}_Cl'] =  neural_net_prediction.flatten() #cl_output 
        
        else:
            neural_net_input = torch.Tensor(inputs['neural_net_input_extrap'])
            neural_net_prediction = neural_net(neural_net_input).detach().numpy()
            outputs['Cl'] =  neural_net_prediction.flatten() #cl_output 
        
    
    def compute_derivatives(self, inputs, derivatives):
        neural_net = self.parameters['neural_net']
        prefix = self.parameters['prefix']

        if prefix:
            neural_net_input = torch.Tensor(inputs[f'{prefix}_neural_net_input_extrap'])
            first_derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input).detach().numpy()
            derivatives[f'{prefix}_Cl', f'{prefix}_neural_net_input_extrap'] =  first_derivative_numpy#  scipy.linalg.block_diag(*derivatives_list)

        else:
            neural_net_input = torch.Tensor(inputs['neural_net_input_extrap'])
            first_derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input).detach().numpy()
            derivatives['Cl', 'neural_net_input_extrap'] =  first_derivative_numpy#  scipy.linalg.block_diag(*derivatives_list)


class CdModel(csdl.CustomExplicitOperation):
    def initialize(self):
        # The neural net will be a pre-trained model
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('neural_net')
        self.parameters.declare('prefix', types=str, default=None, allow_none=True)


    def define(self):
        num_nodes = self.parameters['num_nodes']
        prefix = self.parameters['prefix']

        if prefix:
            self.add_input(f'{prefix}_neural_net_input_extrap', shape=(num_nodes, 3))
            self.add_output(f'{prefix}_Cd', shape=(num_nodes, ))

            self.declare_derivatives(f'{prefix}_Cd', f'{prefix}_neural_net_input_extrap')

        else:
            self.add_input('neural_net_input_extrap', shape=(num_nodes, 3))
            self.add_output('Cd', shape=(num_nodes, ))

            self.declare_derivatives('Cd', 'neural_net_input_extrap')


    
    def compute(self, inputs, outputs):
        neural_net = self.parameters['neural_net']     
        prefix = self.parameters['prefix']

        if prefix:
            neural_net_input = torch.Tensor(inputs[f'{prefix}_neural_net_input_extrap'])
            neural_net_prediction = neural_net(neural_net_input).detach().numpy()
            outputs[f'{prefix}_Cd'] =  neural_net_prediction.flatten() #cl_output 
        
        else:
            neural_net_input = torch.Tensor(inputs['neural_net_input_extrap'])
            neural_net_prediction = neural_net(neural_net_input).detach().numpy()
            outputs['Cd'] =  neural_net_prediction.flatten() #cl_output 
        
    
    def compute_derivatives(self, inputs, derivatives):
        neural_net = self.parameters['neural_net']
        prefix = self.parameters['prefix']

        if prefix:
            neural_net_input = torch.Tensor(inputs[f'{prefix}_neural_net_input_extrap'])
            first_derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input).detach().numpy()
            derivatives[f'{prefix}_Cd', f'{prefix}_neural_net_input_extrap'] =  first_derivative_numpy#  scipy.linalg.block_diag(*derivatives_list)

        else:
            neural_net_input = torch.Tensor(inputs['neural_net_input_extrap'])
            first_derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input).detach().numpy()
            derivatives['Cd', 'neural_net_input_extrap'] =  first_derivative_numpy#  scipy.linalg.block_diag(*derivatives_list)