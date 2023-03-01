import numpy as np
import vedo


class CaddeeVedoContainer():

    def __init__(self, **kwargs):
        """
        initialize a vedo Plotter instance to plot to frame.
        all parameters gets passed into vedo Plotter initialization
        """

        self.plotter = vedo.Plotter(offscreen=True, **kwargs)

        self.actors = []

    def add_point(self, data, **kwargs):
        """
        add a vedo Points instance to add to Plotter instance where data is the point data.
        all parameters gets passed into vedo Point initialization
        """
        self.actors.append(vedo.Points(data, **kwargs))

    def add_sphere(self, **kwargs):
        self.actors.append(vedo.Sphere(**kwargs))

    def add_arrow(self, **kwargs):
        self.actors.append(vedo.Arrow(**kwargs))

    def add_mesh_surface(
            self,
            surface_pts,
            add_to_plotter=False,
            smooth=True,
            wireframe=False,
            fillHoles=False,
            **kwargs):
        """
        given control_points for a surface, create mesh of surface.


        Parameters:
        -----------
            surface_pts: np.ndarray
                surface points with shape (nx,ny,3). points must be ordered such that:

                surface_pts[i,j] is connected to:
                - surface_pts[i+1,j]
                - surface_pts[i-1,j]
                - surface_pts[i,j+1]
                - surface_pts[i,j-1]
        """
        shape = surface_pts.shape
        nx = shape[0]
        ny = shape[1]

        mesh_points = []
        mesh_connections = []
        for i in range(nx):
            for j in range(ny):
                mesh_points.append(list(surface_pts[i, j]))

                c1 = find_flattened_index(i, j, nx, ny)
                c2 = find_flattened_index(i-1, j, nx, ny)
                c3 = find_flattened_index(i, j+1, nx, ny)

                if (c2 is not None) and (c3 is not None):
                    mesh_connections.append([c1, c2, c3])

                # print([c1, c2, c3])
                c1 = find_flattened_index(i, j, nx, ny)
                c2 = find_flattened_index(i+1, j, nx, ny)
                c3 = find_flattened_index(i, j-1, nx, ny)

                if (c2 is not None) and (c3 is not None):
                    mesh_connections.append([c1, c2, c3])

        mesh_data = [mesh_points, mesh_connections]
        # print(kwargs)
        kwargs_new = {}
        if 'c' in kwargs:
            kwargs_new['c'] = kwargs['c']
        if 'alpha' in kwargs:
            kwargs_new['alpha'] = kwargs['alpha']

        vedo_mesh = vedo.Mesh(mesh_data, **kwargs_new)

        if 'contour_map' in kwargs:
            # lut_table = [
            #     #value, color,   alpha, category_label
            #     (0.0, kwargs['c']),
            #     (1.0, 'red'),
            # ]
            lut_table = [
                #value, color,   alpha, category_label
                (0.0, 'blue'),
                (0.99, 'green'),
                (1.0, 'red'),
            ]
            lut = vedo.buildLUT(lut_table, vmin=0.0, vmax=1.0)

            color_map = kwargs['contour_map'].flatten()
            # for i in range(nx):
            #     for j in range(ny):
            #         color_map.append(kwargs['contour_map'][i,j])
            # nv = vedo_mesh.points()   # nr. of cells
            # scals = range(nv)
            # vedo_mesh.cmap('hot', color_map).addScalarBar()
            vedo_mesh.cmap('coolwarm', color_map, vmin=-0.5, vmax=0.0).addScalarBar()

            # vedo_mesh.cmap('hot', color_map, vmin = 0.0, vmax = 1.0).addScalarBar()

        if 'rotateY' in kwargs:
            vedo_mesh.rotateY(kwargs['rotateY'][0], around=kwargs['rotateY'][1])
        if 'rotateX' in kwargs:
            vedo_mesh.rotateX(kwargs['rotateX'][0], around=kwargs['rotateX'][1])
        if 'rotateZ' in kwargs:
            vedo_mesh.rotateZ(kwargs['rotateZ'][0], around=kwargs['rotateZ'][1])

        if add_to_plotter:
            self.actors.append(vedo_mesh)
        return vedo_mesh

    def add_geo_ctrl_pts(self, geometry_control_points, ctrl_pts_metadata):
        """
        given control_points and ctrl_pts_metadata, adds all meshes to actors atribute

        Parameters:
        -----------
            geometry_control_points: np.ndarray
                'ConceptModel.GeometryModel.MeshEvaluationModel.geometry_control_points' array value.
            ctrl_pts_metadata: dict()
                <IMPORTANT> a list with format:

                [{surface0_name: starting_geometry_index, nx, ny, plot_kwargs}, 
                 {surface1_name: starting_geometry_index, nx, ny, plot_kwargs},

                ....

                 {surfacen_name: starting_geometry_index, nx, ny, plot_kwargs}]
            kwargs are keyword arguments to instantiate vedo.Mesh object.
        """
        all_mesh_points = geometry_control_points.reshape(-1, 3)
        surface_meshes_to_merge = []
        for surface_name, surface_metadata in ctrl_pts_metadata.items():
            # pull the point data of each surface from geo_ctrl_pts
            start_ind = surface_metadata[0]
            nx = surface_metadata[1]
            ny = surface_metadata[2]
            kwargs = surface_metadata[3]

            shape = (nx, ny, 3)
            end_ind = start_ind + int(nx*ny)

            # pts_surface is array with shape (nx,ny,3)
            pts_surface = all_mesh_points[start_ind:end_ind].reshape(shape)

            # we know the ordered data of pts_surface so extract ctrl_pts_surface
            surface_meshes_to_merge.append(
                self.add_mesh_surface(
                    pts_surface,
                    add_to_plotter=False,
                    **kwargs,
                )
            )
        self.actors.extend(surface_meshes_to_merge)
        # all_meshes = vedo.merge(surface_meshes_to_merge)
        # self.actors.append(all_meshes)

    def draw_to_axes(self, axes, alpha=1.0, **kwargs):
        """
        draws the 3D Plotter instance to axes.
        """
        self.plotter.show(self.actors, interactive=False, **kwargs)
        image_data = self.plotter.screenshot(asarray=True)
        axes.imshow(image_data, alpha=alpha)
        axes.axis('off')

    def show_interactive(self):
        """
        shows plotter object with interactiviy. ONLY FOR DEBUGGING.
        """
        plotter_interactive = vedo.Plotter(axes=1)
        plotter_interactive.show(self.actors)

def find_flattened_index(i, j, ni, nj):
    if (i == -1) or (i == ni):
        return None
    if (j == nj) or (j == -1):
        return None
    return i*nj + j