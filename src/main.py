# SPDX-FileCopyrightText: 2024 Henrik Jakob jakob@ibb.uni-stuttgart.de
#
# SPDX-License-Identifier: MIT

# import os
# os.environ["DUNE_LOG_LEVEL"] = "debug"
# os.environ["DUNE_SAVE_BUILD"] = "console"

from pathlib import Path
from time import process_time

import ikarus as iks
import ikarus.assembler
import ikarus.dirichlet_values
import ikarus.finite_elements
import numpy as np
import pandas as pd
import pyvista as pv
import scipy as sp


from dune.iga import IGAGrid, IGAGridType
from dune.iga import reader as readeriga


from dune.iga.basis import Nurbs, Power
from fix_dirichlet import fixDofFunction
from nurbs_basis import globalBasis
from tabulate import tabulate


LAMBDA_LOAD = 1.0
THICKNESS = 0.1  # 10 cm
E_MOD = 1000
NU = 0.0




def run_simulation(deg: int, refine: int, testing=False):
    reader = {
        "reader": readeriga.json,
        "file_path": "input/quarter_plate.ibra",
        "trim": True,
        "degree_elevate": (deg - 1, deg - 1),
    }

    gridView = IGAGrid(reader, dimgrid=2, dimworld=2, gridType=IGAGridType.Default)
    gridView.hierarchicalGrid.globalRefine(1)

    basis = globalBasis(gridView, Power(Nurbs(), 2))
    flatBasis = basis.flat()

    ## Define Load
    def vL(x, lambdaVal):
        return np.array([0.0, 1.0])

    ## Define Dirichlet Boundary Conditions
    dirichletValues = iks.dirichletValues(flatBasis)
    dirichletValues.fixDOFs(fixDofFunction)

    ## Create Elements
    vLoad = iks.finite_elements.volumeLoad3D(vL)
    klShell = iks.finite_elements.kirchhoffLoveShell(
        youngs_modulus=E_MOD, nu=NU, thickness=THICKNESS
    )
    fes = []
    for e in gridView.elements:
        fes.append(iks.finite_elements.makeFE(basis, klShell, vLoad))
        fes[-1].bind(e)

    assembler = iks.assembler.sparseFlatAssembler(fes, dirichletValues)

    print(
        f"Size of full System: {flatBasis.dimension}, size of red. System: {assembler.reducedSize()}"
    )

    ## Solve non-linear Kirchhol-Love-Shell problem
    def assemble(dRed_):
        req = ikarus.FERequirements()
        req.addAffordance(iks.ScalarAffordances.mechanicalPotentialEnergy)
        lambdaLoad = iks.ValueWrapper(LAMBDA_LOAD)
        req.insertParameter(iks.FEParameter.loadfactor, lambdaLoad)
        dFull = assembler.createFullVector(dRed_)
        req.insertGlobalSolution(iks.FESolutions.displacement, dFull)
        r = assembler.getReducedVector(req)
        k = assembler.getReducedMatrix(req)
        return [r, k]

    lambdaLoad = iks.ValueWrapper(LAMBDA_LOAD)

    def energy(dRedInput):
        reqL = ikarus.FERequirements()
        reqL.addAffordance(iks.ScalarAffordances.mechanicalPotentialEnergy)
        reqL.insertParameter(iks.FEParameter.loadfactor, lambdaLoad)

        dBig = assembler.createFullVector(dRedInput)
        reqL.insertGlobalSolution(iks.FESolutions.displacement, dBig)
        return assembler.getScalar(reqL)

    def gradient(dRedInput):
        reqL = ikarus.FERequirements()
        reqL.addAffordance(iks.VectorAffordances.forces)
        reqL.insertParameter(iks.FEParameter.loadfactor, lambdaLoad)

        dBig = assembler.createFullVector(dRedInput)
        reqL.insertGlobalSolution(iks.FESolutions.displacement, dBig)
        return assembler.getReducedVector(reqL)

    def hess(dRedInput):
        reqL = ikarus.FERequirements()
        reqL.addAffordance(iks.MatrixAffordances.stiffness)
        reqL.insertParameter(iks.FEParameter.loadfactor, lambdaLoad)

        dBig = assembler.createFullVector(dRedInput)
        reqL.insertGlobalSolution(iks.FESolutions.displacement, dBig)
        return assembler.getReducedMatrix(reqL).todense()

    d = np.zeros(assembler.reducedSize())
    res = sp.optimize.minimize(
        energy,
        method="trust-exact",
        x0=d,
        jac=gradient,
        hess=hess,
        tol=1e-8,
        options={"maxiter": 50},
    )
    d = res.x
    iterations = res.nit
    if res.success:
        print(
            f"Solution found after {iterations} iterations, residual norm: {sp.linalg.norm(res.jac)}"
        )
    else:
        print(
            f"Solution not found after {iterations} iterations, residual norm: {sp.linalg.norm(res.jac)}"
        )

    if testing:
        return 0, 0, 0

    dFull = assembler.createFullVector(d)
    dispFunc = flatBasis.asFunction(dFull)

    vtkWriter = gridView.trimmedVtkWriter()
    vtkWriter.addPointData(dispFunc, name="displacement")
    vtkWriter.write(name=f"{output_folder}/result_d{deg}_r{refine}")

    # Do some postprocessing with pyVista
    mesh = pv.UnstructuredGrid(f"{output_folder}/result_d{deg}_r{refine}.vtu")

    # nodal displacements in z-Direction
    disp_z = mesh["displacement"][:, 2]

    max_d = np.max(disp_z)
    print(f"Max d: {max_d}")

    return max_d, iterations, assembler.reducedSize()


def plot(filename):
    # Postprocessing with pyVista (doesnt seem to work within devcontainer)

    mesh = pv.UnstructuredGrid(filename)
    plotter = pv.Plotter(off_screen=True)
    plotter.view_xy()
    plotter.add_mesh(mesh, scalars="displacement", component=2, show_edges=True)
    # plotter.show()
    plotter.screenshot(f"{output_folder}/displacement.png", transparent_background=True)


if __name__ == "__main__":
    output_folder = Path.cwd() / "output"

    if not output_folder.exists():
        Path.mkdir(output_folder)

    # degree: 2 to 4 (2 = quadratic) We need at least 2 for continuity
    # refine: 3 to 6

    data = []
    for i in range(2, 3):
        for j in range(3, 4):
            t1 = process_time()
            max_d, iterations, dofs = run_simulation(deg=i, refine=j)
            data.append((i, j, max_d, iterations, dofs, process_time() - t1))

    df = pd.DataFrame(
        data,
        columns=["Degree", "Refinement", "max d", "iterations", "DOFs", "Compute time"],
    )
    print(
        tabulate(
            df,
            headers="keys",
            tablefmt="psql",
            floatfmt=("g", "g", "g", "10.10f", "g", "g"),
        )
    )
