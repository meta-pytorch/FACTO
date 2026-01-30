# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.inputgen.argument.type import ArgType
from facto.inputgen.specs.model import ConstraintProducer as cp, InPosArg, Spec
from facto.inputgen.utils.config import TensorConfig


class TestArgumentTupleGenerator(unittest.TestCase):
    def verify_generator_output(self, generator: ArgumentTupleGenerator):
        for posargs, inkwargs, outargs in generator.gen():
            self.assertEqual(len(posargs), 3)
            self.assertEqual(inkwargs, {})
            self.assertEqual(outargs, {})
            t = posargs[0]
            dim = posargs[1]
            tl = posargs[2]
            self.assertTrue(isinstance(t, torch.Tensor))
            self.assertTrue(isinstance(dim, int))
            self.assertTrue(isinstance(tl, list))
            if t.dim() == 0:
                self.assertTrue(dim in [-1, 0])
            else:
                self.assertTrue(dim >= -t.dim())
                self.assertTrue(dim < t.dim())

    def get_spec(self) -> Spec:
        return Spec(
            op="test_size",  # (Tensor self, int dim) -> int
            inspec=[
                InPosArg(ArgType.Tensor, name="self"),
                InPosArg(
                    ArgType.Dim,
                    name="dim",
                    deps=[0],
                    constraints=[
                        cp.Value.Ge(
                            lambda deps: -deps[0].dim() if deps[0].dim() > 0 else None
                        ),
                        cp.Value.Ge(lambda deps: -1 if deps[0].dim() == 0 else None),
                        cp.Value.Le(
                            lambda deps: (
                                deps[0].dim() - 1 if deps[0].dim() > 0 else None
                            )
                        ),
                        cp.Value.Le(lambda deps: 0 if deps[0].dim() == 0 else None),
                    ],
                ),
                InPosArg(ArgType.TensorList, name="tensor_list"),
            ],
            outspec=[],
        )                        

    def test_gen_config(self):
        self.verify_generator_output(ArgumentTupleGenerator(self.get_spec(), TensorConfig()))

    def test_gen_no_config(self):
        self.verify_generator_output(ArgumentTupleGenerator(self.get_spec()))

if __name__ == "__main__":
    unittest.main()
