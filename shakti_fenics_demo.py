import numpy as np
import dolfin as df

df.parameters['form_compiler']['quadrature_degree'] = 4
def Max(a, b): return (a+b+abs(a-b))/2.

# define some useful constants
spy = 60**2*24*365
L = 100000.
W = 20000.
n = 3.0
g = 9.81
La = 3.35e5
G = 0.042
nu = 1.787e-6
omega = 1e-3
e_v = 1e-3
ct = 7.5e-8
cw = 4.22e3

rho_i = 917.
rho_w = 1000.0

h_r = df.Constant(0.1)
l_r = df.Constant(2.0)
A = df.Constant(5.25e-25)

# These will get changed, they're just placeholders
dt_float = 1e-5*spy
dt = df.Constant(dt_float)

# Load a fenics mesh
mesh = df.Mesh('ice_sheet.xml')

# Define facet normals
nhat = df.FacetNormal(mesh)

# Define which facets should be subject to the dirichlet condition
edgefunction = df.MeshFunction('size_t',mesh,1)
for f in df.facets(mesh):
    if df.near(f.midpoint().x(),0):
        edgefunction[f] = 1
ds = df.ds(subdomain_data=edgefunction)

# Define FEM function spaces - CG (i.e. nodal) for head
E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1)
Q_cg = df.FunctionSpace(mesh,E_cg)

# DG for conductivity and cavity size
E_dg = df.FiniteElement("DG",mesh.ufl_cell(),0)
Q_dg = df.FunctionSpace(mesh,E_dg)

# Make a mixed function space so that we can solve all together
E = df.MixedElement([E_cg,E_dg,E_dg])
V = df.FunctionSpace(mesh,E)

# Helper functions for mapping between the big vector containing all variables and individual components
assigner_inv = df.FunctionAssigner([Q_cg,Q_dg,Q_dg],V)
assigner     = df.FunctionAssigner(V,[Q_cg,Q_dg,Q_dg])

# The bed elevation
class B_ex(df.UserExpression):
    def eval(self,values,x):
        values[0] = 0

B = df.interpolate(B_ex(degree=1),Q_cg)

# The ice thickness
class H_ex(df.UserExpression):
    def eval(self,values,x):
        c = 1e5/1500**2
        values[0] = np.sqrt(x[0]/c) + 1
        

H = df.interpolate(H_ex(degree=1),Q_cg)

# The surface elevation
S = H + B

m = df.Function(Q_dg)
m.vector()[:] = 16./spy

# Uncomment this for moulin source, but must be done after initialization, otherwise it won't converge
#m = df.Function(Q_dg)
#m.vector()[:] = 1e-9
#for i in range(50):
#    j = np.random.randint(0,len(m.vector().get_local()))
#    m.vector()[j] = 30.0/400000.

# Define solution vector
U = df.Function(V)
Psi = df.TestFunction(V)
dU = df.TrialFunction(V)

# Split into head, cavity size, diffusivity
h_w,h,K = df.split(U)
xsi,psi,w = df.split(Psi)

# Initialize time dependent quantities/initial guesses
h0_w = df.Function(Q_cg)
h0_w.vector()[:] = B.vector()[:]+rho_i/rho_w*0.01*H.vector()[:]
h0 = df.Function(Q_dg)
h0.vector()[:] = 1e-2
K0 = df.Function(Q_dg)
K0.vector()[:] = 1e-2

P_0 = rho_i*g*H
P_w = rho_w*g*(h_w - B)
N = P_0 - P_w

# shear stress and sliding law
tau_b = rho_i*g*H*S.dx(0)
beta2 = 1e5
u_b = tau_b/(N*beta2)

# Flux
q = -K*df.grad(h_w)

# Opening rate (there's a better way to do this, see my subglacial hydrology paper from 2020)
O = Max(u_b*(h_r - h)/l_r,0)

# Melt rate
M = 1./La*(G + abs(rho_i*g*H*S.dx(0)*u_b) - rho_w*g*df.dot(q,df.grad(h_w)) - ct*cw*rho_w*df.dot(q,df.grad(P_w)))

# Closing rate 
C = A*h*abs(N)**(n-1)*N

# Helper function to scale the melt rate
seasonality = df.Constant(1.0)

# Reynolds number
Re = K*((df.dot(df.grad(h_w),df.grad(h_w))+1e-10)**0.5)/nu

# Conductivity form
R_K = w*(12*nu*(1+omega*Re)*K - abs(h)**3*g)*df.dx

# Cavity change form
R_h = ((h - h0)/dt - O - M/rho_i + C)*psi*df.dx 

# Potential form
R_hw = (e_v*(h_w - h0_w)/dt*xsi + df.dot(df.grad(xsi),K*df.grad(h_w)) + O*xsi - (1/rho_w - 1./rho_i)*M*xsi - C*xsi - seasonality*m*xsi)*df.dx# - xsi*df.dot(K*df.grad(h_w),nhat)*ds(1)

# Add individual forms to get the total one
R = R_K + R_h + R_hw

# Jacobian via symbolic differentiation
J = df.derivative(R,U,dU)

# Boundary dirichlet BC
bc = df.DirichletBC(V.sub(0),df.project(B),edgefunction,1)

# Nonlinear Problem
problem = df.NonlinearVariationalProblem(R,U,J=J,bcs=[bc])
solver = df.NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver'] = 'newton'
solver.parameters['newton_solver']['relaxation_parameter'] = 0.7
solver.parameters['newton_solver']['relative_tolerance'] = 1e-3
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-3
solver.parameters['newton_solver']['error_on_nonconvergence'] = True
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.parameters['newton_solver']['maximum_iterations'] = 30
solver.parameters['newton_solver']['report'] = True

# Initialize solution vector from initial guesses
assigner.assign(U,[h0_w,h0,K0])

# Uncomment if you want to restart from a solution saved as 'steady.xml' via df.File('steady.xml') << U
#df.File('steady.xml') >> U
#assigner_inv.assign([h0_w,h0,K0],U)

# Define start and end times
t = 0
t_end = 10.0*spy

# Max time step
dt_max = 0.01*spy

# Where to save variables
results_dir = 'shakti_test/'
hfile = df.File(results_dir+'h.pvd')
hwfile = df.File(results_dir+'hw.pvd')
qfile = df.File(results_dir+'q.pvd')
qtemp = df.project(q)
Kfile = df.File(results_dir+'K.pvd')
mfile = df.File(results_dir+'m.pvd')
mtemp = df.Function(Q_dg)
Nfile = df.File(results_dir+'N.pvd')
Ntemp = df.Function(Q_cg)
ubfile = df.File(results_dir+'ub.pvd')
ubtemp = df.Function(Q_cg)

# Loop through time
while t<t_end:
    try:
        tt = (t/spy)%1
        # Uncomment this for example of seasonally varying source
        if tt>0.4 and tt<0.7:
            m.vector()[:] = ((-35*np.cos(2*np.pi/0.3*(tt-0.4)) + 35)/spy)
        else:
            m.vector()[:] = (0.1/spy)
        print(m(0,0),dt_float/spy,tt)

        # Solution 
        assigner.assign(U,[h0_w,h0,K0])
        solver.solve()

        # If converged, increase time step
        dt_float = min(1.1*dt_float,dt_max)
        dt.assign(dt_float)
        assigner_inv.assign([h0_w,h0,K0],U)

        # Save variables
        hfile << (h0,t)
        hwfile << (h0_w,t)
        qq = df.project(q)
        qtemp.vector()[:] = qq.vector()[:]
        qfile << (qtemp,t)
        NN = df.project(N)
        Ntemp.vector()[:] = NN.vector()[:]
        Nfile << (Ntemp,t)
        ubb = df.project(u_b)
        ubtemp.vector()[:] = ubb.vector()[:]*spy
        ubfile << (ubtemp,t)
        Kfile << (K0,t)
        t+=dt_float

    except RuntimeError:
        #If solver fails to converge, reduce the time step and try again 
        dt_float/=2.
        dt.assign(dt_float)
        print('convergence failed, reducing time step and trying again')

