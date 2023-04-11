import csdl
import numpy as np 
class BsplineComp(csdl.CustomExplicitOperation):
    """
    General function to translate from control points to actual points
    using a b-spline representation.
    """

    def initialize(self):
        self.parameters.declare('num_pt', types=int)
        self.parameters.declare('num_cp', types=int)
        self.parameters.declare('jac')
        self.parameters.declare('in_name', types=str)
        self.parameters.declare('out_name', types=str)

    def define(self):
        num_pt = self.parameters['num_pt']
        num_cp = self.parameters['num_cp']
        jac = self.parameters['jac']
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']

        self.add_input(in_name, shape=num_cp)
        self.add_output(out_name, shape=num_pt)

        jac = self.parameters['jac'].tocoo()

        self.declare_derivatives(out_name, in_name, val=jac.data, rows=jac.row, cols=jac.col)

    def compute(self, inputs, outputs):
        num_pt = self.parameters['num_pt']
        num_cp = self.parameters['num_cp']
        jac = self.parameters['jac']
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']

        # interp_range = np.linspace(0,1,num_pt)
        # interp_points_x = np.linspace(0,1,num_cp)
        # interp_points_y = inputs[in_name]

        # mat = np.array([[interp_points_x[0]**3,interp_points_x[0]**2,interp_points_x[0],1],
        #                 [interp_points_x[1]**3,interp_points_x[1]**2,interp_points_x[1],1],
        #                 [interp_points_x[2]**3,interp_points_x[2]**2,interp_points_x[2],1],
        #                 [interp_points_x[3]**3,interp_points_x[3]**2,interp_points_x[3],1]])
                        
        # label = np.array([[interp_points_y[0],interp_points_y[1],interp_points_y[2],interp_points_y[3]]])

        # sol = np.linalg.solve(mat,label.T)
        # print(sol)

        # twist = sol[0]*interp_range**3 + sol[1]*interp_range**2 + sol[2]*interp_range + sol[3]

        # print('STUFF',jac.shape)
        # print(inputs[in_name])
        outputs[out_name] = jac * inputs[in_name]