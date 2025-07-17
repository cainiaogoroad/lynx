# Copyright 2023 The Lynx Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

workspace(name = "lynx")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/bazel:brpc.bzl", "load_brpc")
load("//third_party/bazel:pybind.bzl", "load_pybind")

# BRPC dependencies
load_brpc()

load_pybind()


load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies(register_preinstalled_tools = False)

load("@rules_perl//perl:deps.bzl", "perl_register_toolchains")

perl_register_toolchains()



load("@pybind11_bazel//:python_configure.bzl", "python_configure")
# This is required for setting up the linkopts for -lpython.q
python_configure(
    name = "local_config_python",
    python_version = "3",  # required to use `python3-config`
)
