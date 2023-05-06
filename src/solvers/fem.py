from abc import ABC, abstractmethod
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.utils.quadrules import gauss_lobatto_jacobi_quadrature1D


def get_sol(u, bases):
    assert len(u) == len(bases)
    return lambda x: np.sum([u[i] * bases[i](x) for i in range(len(bases))], axis=0)

def get_der(u, bases):
    assert len(u) == len(bases)
    return lambda x: np.sum([u[i] * bases[i].deriv(1)(x) for i in range(len(bases))], axis=0)

def plot_bases(S, bases):

    xs = np.linspace(-1, 1, 100)
    us = []
    # MODIFY: Combine these two loops
    sols = [get_sol(S[:, j], bases) for j in range(S.shape[1])]
    for idx, sol in enumerate(sols):
        us.append(sol(xs))

    df = pd.DataFrame({
        'x': np.concatenate([xs] * len(us)),
        'u_re': np.concatenate([u.real for u in us]),
        'u_im': np.concatenate([u.imag for u in us]),
        'j': np.concatenate([np.repeat(j, len(xs)) for j in range(len(sols))]),
    })

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20, 10))
    sns.lineplot(
        data=df,
        x='x',
        y='u_re',
        hue='j',
        ax=axs[0],
    )
    sns.lineplot(
        data=df,
        x='x',
        y='u_im',
        hue='j',
        ax=axs[1],
    )

def H1_error(e: Callable, e_x: Callable, solver) -> float:

    u2 = lambda x: abs(e(x)) ** 2
    ux2 = lambda x: abs(e_x(x)) ** 2

    err_u = np.sqrt(solver.intg(u2))
    err_u_x = np.sqrt(solver.intg(ux2))

    return err_u + err_u_x, err_u, err_u_x

class DerivableFunction:
    def __init__(self, func: Callable, derivs: list =None):
        self.func = func
        self.derivs = derivs if derivs else None

    def deriv(self, i: int):
        assert self.derivs and i <= len(self.derivs)
        if i == 0:
            return self.func
        else:
            return self.derivs[i - 1]

    def __call__(self, x: float):
        return self.func(x)

class FEBasisFunctions:
    def __init__(self, K: int, a: float =-1, b: float =+1, dtype=torch.Tensor, device=None):
        self.dtype = dtype
        self.h = (b - a) / K
        self.K = K
        self.a = a
        self.b = b

        if self.dtype is torch.Tensor:
            self.device = device if device else torch.device('cpu')
            self.x = torch.linspace(a, b, K + 1).to(self.device)
        elif self.dtype is np.ndarray:
            self.x = np.linspace(a, b, K + 1)

    def __call__(self):
        """Returns the finite elements in a list.
        """

        elements = []
        for k in range(self.K + 1):
            element = DerivableFunction(self.phi(k), [self.phi_x(k), self.phi_xx(k)])

            # Define the local support domain of the element
            xm = self.x[k]
            xl = self.x[k - 1] if k != 0 else xm
            xr = self.x[k + 1] if k != self.K else xm
            element.domain_ = (xl, xm, xr)

            elements.append(element)

        return elements

    def phi(self, i: int) -> Callable:
        """Returns the basis functions of the V_N subspace

        Args:
            i (int): Index of the basis function.

        Returns:
            Callable: The basis function.
        """

        xm = self.x[i]
        xl = self.x[i - 1] if i != 0 else xm
        xr = self.x[i + 1] if i != self.K else xm

        if self.dtype is torch.Tensor:
            step = lambda x: torch.heaviside(x - xl, torch.tensor([1.], device=self.device))\
                - torch.heaviside(x - xr, torch.tensor([0.], device=self.device))
            return lambda x: step(x) * (1 - torch.abs((x - xm)) / self.h)

        elif self.dtype is np.ndarray:
            step = lambda x: np.heaviside(x - xl, 1)\
                - np.heaviside(x - xr, 0)
            return lambda x: step(x) * (1 - np.abs((x - xm)) / self.h)

    def phi_x(self, i: int) -> Callable:
        """Returns the derivative of the basis function.

        Args:
            i (int): Index of the basis function.

        Returns:
            Callable: The derivative of the basis function
        """

        xm = self.x[i]
        xl = self.x[i - 1] if i != 0 else xm
        xr = self.x[i + 1] if i != self.K else xm

        if self.dtype is torch.Tensor:
            step = lambda x: torch.heaviside(x - xl, torch.tensor([1.], device=self.device))\
                - torch.heaviside(x - xr, torch.tensor([0.], device=self.device))
        elif self.dtype is np.ndarray:
            step = lambda x: np.heaviside(x - xl, 1)\
                - np.heaviside(x - xr, 0)

        if i == 0:
            return lambda x: step(x) * -(1 / self.h) * (+1)
        elif i == self.K:
            return lambda x: step(x) * -(1 / self.h) * (-1)
        else:
            if self.dtype is torch.Tensor:
                return lambda x: step(x) * -(1 / self.h) * torch.sign(x - xm)
            elif self.dtype is np.ndarray:
                return lambda x: step(x) * -(1 / self.h) * np.sign(x - xm)

    def phi_xx(self, i: int) -> Callable:
        """Returns the second derivative of the basis functions of the V_N subspace

        Args:
            i (int): Index of the basis function.

        Returns:
            Callable: The basis function.
        """

        return lambda x: 0

    def intphi(self, i: int, j: int) -> float:
        """Calculates int(phi_i * phi_j) over the domain, 0 <= i, j <= N.

        Args:
            i (int): Index of the basis function.
            j (int): Index of the test function.

        Returns:
            float: The value of the integral.
        """

        if i == j:
            if i == 0 or i == self.K:
                return self.h / 3
            else:
                return self.h * 2 / 3
        elif abs(i - j) == 1:
            return self.h / 6
        else:
            return 0

    def intphi_x(self, i: int, j: int) -> float:
        """Calculates int(phi_x_i * phi_x_j) over the domain, 0 <= i, j <= N.

        Args:
            i (int): Index of the basis function.
            j (int): Index of the test function.

        Returns:
            float: The value of the integral.
        """

        if i == j:
            if i == 0 or i == self.K:
                return 1 / self.h
            else:
                return 2 / self.h
        elif abs(i - j) == 1:
            return -1 / self.h
        else:
            return 0

class FEMSolver1D(ABC):
    """Base class for 1D FEM solvers."""
    def __init__(self, a: float, b: float, *, N: int = 50, N_quad: int = None):

        """Initializing the parameters

        Args:
            a (float): Left boundary.
            b (float): Right boundary.
            N (int, optional): Number of discretization points. Defaults to 50.
            N_quad (int, optional): Number of quadrature points for int(f * phi).
        """

        # Store the parameters
        self.a, self.b = a, b
        self.N = N
        self.h = (b - a) / N

        # Get basis functions
        self.FE = FEBasisFunctions(N, a, b, dtype=np.ndarray)
        self.bases = self.FE()
        if not N_quad:
            self.N_quad = min(self.N * 10, 1000)
        else:
            self.N_quad = N_quad

        # Initialize coefficients
        self.x = np.linspace(a, b, N + 1)
        self.A = np.zeros((N + 1, N + 1), dtype=complex)
        self.d = np.zeros(N + 1, dtype=complex)

        # Initialize solutions
        self.c = None
        self.sol = None
        self.der = None

        # Get the quadrature points
        self.roots, self.weights = gauss_lobatto_jacobi_quadrature1D(self.N_quad, a, b)
        self.roots, self.weights = self.roots.numpy(), self.weights.numpy()

    def solve(self):
        """Executes the method.
        """

        for i in range(self.N + 1):
            self.d[i] = self.rhs(i)
            self.A[i, i] = self.lhs(i, i)
            if i != 0:
                self.A[i, i - 1] = self.lhs(i, i - 1)
            if i != self.N:
                self.A[i, i + 1] = self.lhs(i, i + 1)

        self.c = np.linalg.solve(self.A, self.d)
        self.sol = lambda x: np.sum(np.array(
            [[self.c[i] * self.bases[i](x)] for i in range(self.N + 1)]
            ), axis=0)
        self.der = lambda x: np.sum(np.array(
            [[self.c[i] * self.bases[i].deriv(1)(x)] for i in range(self.N + 1)]
            ), axis=0)

    @abstractmethod
    def lhs(self, i, j) -> complex:
        """Computes the ij^{th} element of the left-hand-side of the system of the equations.

        Args:
            i (int): Index of the basis function.
            j (int): Index of the test function.

        Returns:
            complex: The left-hand-side of the equation.
        """
        ...

    @abstractmethod
    def rhs(self, j: int) -> complex:
        """Computes the j^{th} element of the right-hand-side of the system of the equations.

        Args:
            j (int): Index of the test function.

        Returns:
            complex: The right-hand-side of the equation.
        """
        ...

    def H1_error(self, u: Callable, u_x: Callable) -> float:
        """Computes the H1 error:
        .. math::
            \\sqrt{||u - u_N||_{L^2}^2 + ||\\frac{du}{dx} - \\frac{du_N}{dx}||_{L^2}^2}

        Args:
            u (Callable): Exact solution
            u_x (Callable): Exact derivative of the solution

        Returns:
            float: H1 error of the solution
            float: L2 norm of the solution
            float: L2 norm of the derivative of the solution
        """

        if not self.sol:
            print(f'Call {self.__class__.__name__}.solve() to find the FEM solution first.')
            return

        u2 = lambda x: abs(u(x) - self.sol(x)) ** 2
        ux2 = lambda x: abs(u_x(x) - self.der(x)) ** 2

        err_u = np.sqrt(self.intg(u2))
        err_u_x = np.sqrt(self.intg(ux2))

        return err_u + err_u_x, err_u, err_u_x

    def intg(self, func: Callable) -> complex:
        """Integrator of the class.

        Args:
            func (Callable): Function to be integrated.

        Returns:
            complex: Integral over the domain (a, b) with N_quad quadrature points.
        """

        # return integrate_1d(f, self.a, self.b, self.weights, self.roots).item()
        return (self.b - self.a) * np.sum(func(self.roots) * self.weights) / 2

    def __call__(self, x: float) -> complex:
        """Returns the solution of the equation.

        Args:
            x (float): Point to evaluate the solution.

        Returns:
            complex: Value of the solution.
        """

        if not self.sol:
            print(f'Call {self.__class__.__name__}.solve() to find the FEM solution first.')
            return
        return self.sol(x), self.der(x)

class Poisson1D(FEMSolver1D):
    # TODO: Generalize to mixed boundary conditions
    """
    Finite Element Method solver class for solving the 1D Poisson equation:
        - (k u_x)_x = f
    in the domain (a, b), with Dirichlet boundary conditions:
        u(a) = 0
        u(b) = 0
    """

    def __init__(self, f: Callable, k: Callable, a: float = -1, b: float = +1,
        N: int = 50, N_quad: int = None):

        """Initializing the parameters

        Args:
            f: Source function.
            k: Diffusion coefficient.
            a: Left boundary.
            b: Right boundary.
            N: Number of discretization points. Defaults to 50.
            N_quad: Number of quadrature points for int(f * phi).
        """

        # Store specific parameters
        self.f = f
        self.k = k

        super().__init__(a=a, b=b, N=N, N_quad=N_quad)

    def lhs(self, i, j) -> complex:
        """Computes the left-hand-side of the system of the equations:
            int(phi_x_i * phi_x_j) - k^2 * int(phi_i * phi_j)
            - 1j * k * (phi_i(a) * phi_j(a) + phi_i(b) * phi_j(b))

        Args:
            i (int): Index of the basis function.
            j (int): Index of the test function.

        Returns:
            complex: The left-hand-side of the equation.
        """

        phi_i: DerivableFunction = self.bases[i]
        phi_j: DerivableFunction = self.bases[j]

        # TODO: Take the integral only on the small subdomain that the function is nonzero
        kphiphi = lambda x: self.k(x) * phi_i.deriv(1)(x) * phi_j.deriv(1)(x)

        return self.intg(kphiphi)

    def rhs(self, j: int) -> complex:
        """Computes the right-hand-side of the system of the equations:
            \int(f * phi_j)

        Args:
            j (int): Index of the test function.

        Returns:
            complex: The right-hand-side of the equation.
        """

        phi_j = self.bases[j]
        fv = lambda x: self.f(x) * phi_j(x)
        intfv = self.intg(fv)

        return intfv

class HelmholtzImpedance1D(FEMSolver1D):
    """
    Finite Element Method solver class for solving the 1D Helmholtz Impendace problem:
        - u_xx - k^2 * u = f
    in the domain (a, b), with impedance boundary conditions:
        - u_x(a) - 1j * k * u(a) = ga
        + u_x(b) - 1j * k * u(b) = gb
    """

    def __init__(self, f: Union[Callable, float], k: float, a: float, b: float, \
        ga: complex, gb: complex, *, N: int = 50, N_quad: int = None):

        """Initializing the parameters

        Args:
            f (function or float): Source function or the coefficient.
            k (float): Equation coefficient.
            a (float): Left boundary.
            b (float): Right boundary.
            ga (complex): Value of the left boundary condition.
            gb (complex): Value of the right boundary condition.
            N (int, optional): Number of discretization points. Defaults to 50.
            N_quad (int, optional): Number of quadrature points for int(f * phi).
        """

        # Store specific parameters
        self.f = f
        self.k = k
        self.ga, self.gb = ga, gb

        super().__init__(a=a, b=b, N=N, N_quad=N_quad)

    def lhs(self, i, j) -> complex:
        """Computes the left-hand-side of the system of the equations:
            int(phi_x_i * phi_x_j) - k^2 * int(phi_i * phi_j)
            - 1j * k * (phi_i(a) * phi_j(a) + phi_i(b) * phi_j(b))

        Args:
            i (int): Index of the basis function.
            j (int): Index of the test function.

        Returns:
            complex: The left-hand-side of the equation.
        """

        phi_i = self.bases[i]
        phi_j = self.bases[j]
        return self.FE.intphi_x(i, j) - self.k ** 2 * self.FE.intphi(i, j)\
            - 1j * self.k * (phi_i(self.a) * phi_j(self.a) + phi_i(self.b) * phi_j(self.b))

    def rhs(self, j: int) -> complex:
        """Computes the right-hand-side of the system of the equations:
            int(f * phi_j) + ga * phi_j(a) + gb * phi_j(b)

        Args:
            j (int): Index of the test function.

        Returns:
            complex: The right-hand-side of the equation.
        """

        phi_j = self.bases[j]

        if isinstance(self.f, Callable):
            fv = lambda x: self.f(x) * phi_j(x)
            intfv = self.intg(fv)
        elif isinstance(self.f, float):
            intfv = self.f * self.h
            if j == 0 or j == self.N:
                intfv = intfv / 2
        else:
            raise ValueError(f'{self.source} is not a valid source type.')

        return intfv + self.ga * phi_j(self.a) + self.gb * phi_j(self.b)

class ParameterizedSolver:
    """Class for using an FEM solver with an affine parameterized uncertainty."""

    def __init__(self, solver: FEMSolver1D, a: Union[list[Callable], list[float]], uc: str):
        #: List of callables that define the uncertainty
        self.a = a
        #: A solver
        self.solver = solver
        #: Name of the uncertainty argument of the solver
        self.uc = uc

    def solve(self, y: np.ndarray):
        # Check shape of the input
        assert y.shape[0] == len(self.a)

        # Solution coefficients
        c = np.zeros(shape=(self.solver.N+1, y.shape[1]), dtype=complex)
        for j in range(y.shape[1]):
            # Generate the uncertainty parameter
            if isinstance(self.a[0], Callable):
                a = lambda x: np.sum([y[:, j][idx] * self.a[idx](x) for idx in range(len(y))])
            elif isinstance(self.a[0], (float, int)):
                # MODIFY: Use matrix multiplication
                a = np.sum([y[:, j][idx] * self.a[idx] for idx in range(len(y))])
            # Solve with the generated parameter
            self.solver.__dict__[self.uc] = a
            self.solver.solve()
            # Store the solution
            c[:, j] = self.solver.c

        return c
