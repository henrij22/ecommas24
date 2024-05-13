# SPDX-FileCopyrightText: 2024 Henrik Jakob jakob@ibb.uni-stuttgart.de
#
# SPDX-License-Identifier: MIT

from dune.iga.basis import defaultGlobalBasis

from dune.generator.generator import SimpleGenerator
from dune.iga.basis import preBasisTypeName
from dune.common.hashit import hashIt


# this workaround is needed as the python interface for dune-functions is a bit to general, so we have to create a Ikarus BasisHandler ourself
def globalBasis(gv, tree):
    generator = SimpleGenerator("BasisHandler", "Ikarus::Python")

    pbfName = preBasisTypeName(tree, gv.cppTypeName)
    element_type = f"Ikarus::BasisHandler<{pbfName}>"
    includes = []
    includes += list(gv.cppIncludes)
    includes += ["dune/iga/nurbsbasis.hh"]
    includes += ["ikarus/python/basis/basis.hh"]

    moduleName = "Basis_" + hashIt(element_type)
    module = generator.load(
        includes=includes, typeName=element_type, moduleName=moduleName
    )
    basis = defaultGlobalBasis(gv, tree)
    return module.BasisHandler(basis)
