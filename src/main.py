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
import ikarus.utils
import numpy as np
import pandas as pd
import pyvista as pv
import scipy as sp


import dune.iga
from dune.iga import IGAGrid, IGAGridType
from dune.iga import reader as readeriga


from dune.iga.basis import Nurbs, Power
from fix_dirichlet import fixDofFunction
from nurbs_basis import globalBasis
from tabulate import tabulate

from analytical import AnalyticalSolution

LAMBDA_LOAD = 1.0
THICKNESS = 0.1  # 10 cm
E_MOD = 1000
NU = 0.0

analyticalSolution = AnalyticalSolution(
    E=E_MOD, nu=NU, Tx=1, R=1, offset=np.array([0, 0])
)


def run_simulation(deg: int, refine: int, testing=False):
    reader = {
        "reader": readeriga.json,
        "file_path": "input/quarter_plate.ibra",
        "trim": True,
        "degree_elevate": (deg - 1, deg - 1)
    }

    gridView = IGAGrid(reader, dimgrid=2, dimworld=2, gridType=IGAGridType.Default)
    # for _ in range(refine):
    gridView.hierarchicalGrid.globalRefine(1)

    dune.iga.registerTrimmerPreferences(targetAccuracy=0.001)

    basis = globalBasis(gridView, Power(Nurbs(), 2))
    flatBasis = basis.flat()

    ## Define Load
    def neumannLoad(x, lambdaVal):
        stresses = analyticalSolution.stressSolution(x)

        # left side
        if x[0] > 4 - 1e-8:
            return np.array([stresses[0], stresses[2]])
        elif x[1] > 4 - 1e-8:
            return np.array([stresses[2], stresses[1]])
        return np.array([0.0, 0.0])

    neumannVertices = np.zeros(gridView.size(2), dtype=bool)

    def loadPredicate(x):
        return abs(x[0]) > 4 - 1e-8 or abs(x[1]) > 4 - 1e-8

    indexSet = gridView.indexSet
    for v in gridView.vertices:
        neumannVertices[indexSet.index(v)] = loadPredicate(v.geometry.center)

    boundaryPatch = dune.iga.boundaryPatch(gridView, neumannVertices)
    nBLoad = iks.finite_elements.neumannBoundaryLoad(boundaryPatch, neumannLoad)

    ## Define Dirichlet Boundary Conditions
    dirichletValues = iks.dirichletValues(flatBasis)
    dirichletValues.fixDOFs(fixDofFunction)

    ## Create Elements
    linearElastic = ikarus.finite_elements.linearElastic(youngs_modulus=E_MOD, nu=NU)

    fes = []
    for e in gridView.elements:
        fes.append(iks.finite_elements.makeFE(basis, linearElastic, nBLoad))
        fes[-1].bind(e)

    assembler = iks.assembler.sparseFlatAssembler(fes, dirichletValues)

    print(
        f"Size of full System: {flatBasis.dimension}, size of red. System: {assembler.reducedSize()}"
    )

    lambdaLoad = iks.ValueWrapper(LAMBDA_LOAD)

    d = np.zeros(len(flatBasis))

    req = ikarus.FERequirements()
    req.addAffordance(iks.ScalarAffordances.mechanicalPotentialEnergy)
    req.insertParameter(iks.FEParameter.loadfactor, lambdaLoad)
    req.insertGlobalSolution(iks.FESolutions.displacement, d)

    K = assembler.getMatrix(req)
    F = assembler.getVector(req)

    d = sp.sparse.linalg.spsolve(K, F)
   
    req.insertGlobalSolution(iks.FESolutions.displacement, d)

    dispFunc = flatBasis.asFunction(d)
    stressFunc = gridView.function(
        lambda e, x: fes[indexSet.index(e)].calculateAt(req, x, "linearStress")[:]
    )

    vtkWriter = gridView.trimmedVtkWriter(0)
    vtkWriter.addPointData(dispFunc, name="displacement")
    vtkWriter.addPointData(stressFunc, name="stress")
    vtkWriter.write(name=f"{output_folder}/result_d{deg}_r{refine}")

    # # Do some postprocessing with pyVista
    # mesh = pv.UnstructuredGrid(f"{output_folder}/result_d{deg}_r{refine}.vtu")

    # # nodal displacements in z-Direction
    # disp_z = mesh["displacement"][:, 1]

    # max_d = np.max(disp_z)
    # print(f"Max d: {max_d}")

    return 0, 0, 0


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
