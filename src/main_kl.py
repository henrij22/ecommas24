# SPDX-FileCopyrightText: 2024 Henrik Jakob jakob@ibb.uni-stuttgart.de
#
# SPDX-License-Identifier: MIT

import os

# os.environ["DUNE_LOG_LEVEL"] = "debug"
# os.environ["DUNE_SAVE_BUILD"] = "console"
os.environ["DUNE_CMAKE_FLAGS"] = "-CMAKE_BUILD_TYPE=Debug"


from pathlib import Path
from time import process_time

import ikarus as iks
import ikarus.assembler
import ikarus.dirichlet_values
import ikarus.finite_elements
import ikarus.utils
import numpy as np
import pandas as pd
import pyvista as pv
import scipy as sp


import dune.iga
from dune.iga import IGAGrid, IGAGridType
from dune.iga import reader as readeriga


from dune.iga.basis import Nurbs, Power
from fix_dirichlet import fixDofFunction2
from nurbs_basis import globalBasis
from tabulate import tabulate

from analytical import AnalyticalSolution

LAMBDA_LOAD = 1.0
THICKNESS = 0.1  # 10 cm
E_MOD = 1000
NU = 0.0


def run_simulation(deg: int, refine: int, testing=False):
    reader = {
        "reader": readeriga.json,
        "file_path": "input/surface-hole.ibra",
        "trim": True,
        "degree_elevate": (deg - 1, deg - 1),
        "post_knot_refinement": (refine, refine),
    }

    gridView = IGAGrid(reader, dimgrid=2, dimworld=3, gridType=IGAGridType.Default)

    basis = globalBasis(gridView, Power(Nurbs(), 3))
    flatBasis = basis.flat()

    dune.iga.registerTrimmerPreferences(targetAccuracy=0.001)

     ## Define Load
    def vL(x, lambdaVal):
        return np.array([0, 0, 2 * THICKNESS**3 * lambdaVal])

    ## Define Dirichlet Boundary Conditions
    dirichletValues = iks.dirichletValues(flatBasis)
    dirichletValues.fixDOFs(fixDofFunction2)

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

    lambdaLoad = iks.ValueWrapper(LAMBDA_LOAD)
    d = np.zeros(flatBasis.dimension)

    req = ikarus.FERequirements()
    req.addAffordance(iks.ScalarAffordances.mechanicalPotentialEnergy)
    req.insertParameter(iks.FEParameter.loadfactor, lambdaLoad)
    req.insertGlobalSolution(iks.FESolutions.displacement, d)

    #K = assembler.getMatrix(req)
    F = assembler.getVector(req)

    # d = sp.sparse.linalg.spsolve(K, F)
    # d = sp.linalg.solve(K, F)

    # # req.insertGlobalSolution(iks.FESolutions.displacement, d)

    # dispFunc = flatBasis.asFunction(d)
    # # stressFunc = gridView.function(
    # #     lambda e, x: fes[indexSet.index(e)].calculateAt(req, x, "linearStress")[:]
    # # )

    # vtkWriter = gridView.trimmedVtkWriter(0)
    # vtkWriter.addPointData(dispFunc, name="displacement")
    # #vtkWriter.addPointData(stressFunc, name="stress")
    # vtkWriter.write(name=f"{output_folder}/result_d{deg}_r{refine}")

    # # Do some postprocessing with pyVista
    # mesh = pv.UnstructuredGrid(f"{output_folder}/result_d{deg}_r{refine}.vtu")

    # # nodal displacements in z-Direction
    # disp_z = mesh["displacement"][:, 1]

    # max_d = np.max(disp_z)
    # print(f"Max d: {max_d}")

    return 0, 0, flatBasis.dimension


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
    for i in range(2, 4):
        for j in range(3, 6):
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
