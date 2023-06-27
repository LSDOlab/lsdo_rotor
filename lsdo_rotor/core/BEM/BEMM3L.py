import m3l
import csdl


class BEMM3L(m3l.ExplicitOperation):
    def initialize(self):
        self.parameters.declare('component')
        self.parameters.declare('mesh')
        self.num_nodes = None

    def compute(self) -> csdl.Model:
        from lsdo_rotor.core.BEM_caddee.BEM_m3l_model import BEMModel
        component = self.parameters.declare('component')
        mesh = self.parameters.declare('mesh')
        num_nodes = self.num_nodes
        prefix = component.parameters['name']
        
        return BEMModel(
            mesh=mesh,
            num_nodes=num_nodes,
            prefix=prefix,
        )
    
    def evaluate_forces_moments(self): 
        csdl_operation = self.compute()

        F_M_operation = m3l.CSDLOperation(name='evaluate_forces_moments', arguments=[], operation_csdl=csdl_operation)
        F = m3l.Variable(name='F', shape=(self.num_nodes, 3), operation=F_M_operation)
        M = m3l.Variable(name='M', shape=(self.num_nodes, 3), operation=F_M_operation)

        return F, M




    
