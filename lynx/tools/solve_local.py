# coding=utf-8
from argparse import ArgumentParser
import os
import json
from subprocess import Popen, PIPE
import time
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_file_path')
    parser.add_argument('dst_file')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--env', default="prod", choices=["prod", "dev"])
    args = parser.parse_args()
    # temp redirect to java script
    java_path = os.path.join(
        os.path.dirname(__file__), 
        "DAG-0317.jar"
        )
    p = Popen(["java","-jar",
        java_path,
        args.input_file_path,
        args.dst_file],
        stdout=PIPE,
        stderr=PIPE)
    stdout, stderr = p.communicate()
    print(stdout)
    print(stderr)
    with open(args.dst_file, "r") as fin:
        json_obj = json.load(fin)
    with open(args.dst_file, "w") as fout:
        json.dump(json_obj, fout, indent=4)
        fout.flush()
    time.sleep(5)