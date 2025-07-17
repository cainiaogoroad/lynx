#-*- coding: utf-8 -*-

import os
import subprocess
import sys
import unittest


class TestBasicCases(unittest.TestCase):

  def setUp(self):
    super(TestBasicCases, self).setUp()
    lynx_root_path = os.path.dirname(
        os.path.abspath(os.path.join(__file__, "../")))
    os.environ["LYNX_ROOT"] = lynx_root_path
    # reboot the cluster
    subprocess.Popen([
        "%s/build/bazel-bin/lynx/csrc/runtime/global_manager/gm_main" %
        lynx_root_path, "--port=18000"
    ])
    subprocess.Popen([
        "%s/build/bazel-bin/lynx/csrc/runtime/local_manager/lm_main" %
        lynx_root_path,
        "--gm_server=127.0.0.1:18000",
        "--port=18001",
    ])

  def tearDown(self):
    super(TestBasicCases, self).tearDown()
    # kill dead processes
    os.system(
        "ps aux | grep -v grep | grep gm_main | awk '{print $2}' | xargs kill -9"
    )
    os.system(
        "ps aux | grep -v grep | grep lm_main | awk '{print $2}' | xargs kill -9"
    )

  def test_connect_cluster(self):
    import lynx
    lynx.connect_cluster(0, 0, 0, 0, 0, "127.0.0.1:18001", True)


if __name__ == "__main__":
  test = unittest.main(exit=False)
  sys.exit(0 if test.result.wasSuccessful() else 1)
