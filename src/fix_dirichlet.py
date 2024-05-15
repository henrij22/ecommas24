# SPDX-FileCopyrightText: 2024 Henrik Jakob jakob@ibb.uni-stuttgart.de
#
# SPDX-License-Identifier: MIT

from io import StringIO
from dune.generator.algorithm import run


fixFunction = """
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/functions/functionspacebases/subentitydofs.hh>
#include <dune/iga/utils/igahelpers.hh>
#include <dune/python/pybind11/eigen.h>
void fixFunction(auto& basis_, auto dirichletFlags_) {
  auto dirichletFlags = dirichletFlags_.template cast<Eigen::Ref<Eigen::VectorX<bool>>>();
  Dune::Functions::forEachUntrimmedBoundaryDOF(
        Dune::Functions::subspaceBasis(basis_, 1), [&](auto &&localIndex, auto &&localView, auto &&intersection) {
          if (std::fabs(intersection.geometry().center()[1]) < 1e-8)
            dirichletFlags[localView.index(localIndex)] = true;
            std::cout << "D: " << dirichletFlags.size() << std::endl;
                                                 std::cout << "I: " << localView.index(localIndex) << std::endl;
        });
    Dune::Functions::forEachUntrimmedBoundaryDOF(
        Dune::Functions::subspaceBasis(basis_, 0), [&](auto &&localIndex, auto &&localView, auto &&intersection) {
          if (std::fabs(intersection.geometry().center()[0]) < 1e-8)
            dirichletFlags[localView.index(localIndex)] = true;
            std::cout << "D: " << dirichletFlags.size() << std::endl;
                                                 std::cout << "I: " << localView.index(localIndex) << std::endl;
        });
}
"""


def fixDofFunction(basis_, dirichletFlags_):
    run("fixFunction", StringIO(fixFunction), basis_, dirichletFlags_)


fixFunction2 = """
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/functions/functionspacebases/subentitydofs.hh>
#include <dune/iga/utils/igahelpers.hh>
#include <dune/python/pybind11/eigen.h>
void fixFunction(auto& basis_, auto dirichletFlags_) {
  auto dirichletFlags = dirichletFlags_.template cast<Eigen::Ref<Eigen::VectorX<bool>>>();
  Dune::Functions::forEachUntrimmedBoundaryDOF(basis_,
                                                 [&](auto&& localIndex, auto&& localView, auto&& intersection) {
                                                 std::cout << "D: " << dirichletFlags.size() << std::endl;
                                                 std::cout << "I: " << localView.index(localIndex) << std::endl;
                                                   dirichletFlags[localView.index(localIndex)] = true;
                                                 });
}
"""


def fixDofFunction2(basis_, dirichletFlags_):
    run("fixFunction", StringIO(fixFunction2), basis_, dirichletFlags_)


### Backup
fixFunction_ = """
#define EIGEN_DEFAULT_TO_ROW_MAJOR 1
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/functions/functionspacebases/subentitydofs.hh>
#include <dune/iga/utils/igahelpers.hh>
#include <dune/python/pybind11/eigen.h>
void fixFunction(auto& basis_, auto dirichletFlags_) {
  auto dirichletFlags = dirichletFlags_.template cast<Eigen::Ref<Eigen::VectorX<bool>>>();
  Dune::Functions::forEachUntrimmedBoundaryDOF(Dune::Functions::subspaceBasis(basis_, 2),
                                                 [&](auto&& localIndex, auto&& localView, auto&& intersection) {
                                                   dirichletFlags[localView.index(localIndex)] = true;
                                                 });
    auto fixEverything = [&](auto&& subBasis_) {
      auto localView       = subBasis_.localView();
      auto seDOFs          = subEntityDOFs(subBasis_);
      const auto& gridView = subBasis_.gridView();
      for (auto&& element : elements(gridView)) {
        localView.bind(element);
        for (const auto& intersection : intersections(gridView, element))
          for (auto localIndex : seDOFs.bind(localView, intersection))
           dirichletFlags[localView.index(localIndex)] = true;
      }
    };
    fixEverything(Dune::Functions::subspaceBasis(basis_, 0));
    fixEverything(Dune::Functions::subspaceBasis(basis_, 1));
}
"""
