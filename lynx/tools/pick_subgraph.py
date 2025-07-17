#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
"""
subgraph_define_file example:
```
digraph G {
rankdir = TB;
compound = true;
label = <<b>module_0205.SyncTensorsGraph.33142.sm_8.0_gpu_after_optimizations<br/>Computation SyncTensorsGraph.33142_spmd</b>>;
labelloc = t;
// Disable the tooltip.  Interestingly, "" doesn't work!
tooltip = " ";
subgraph cluster_3869814064 {
style="rounded,filled,bold"; fillcolor="#f5f5f5"; color="#c2c2c2;"
label = <Fused expression for <b>loop_convert_fusion.80</b><br/>loop fusion<br/>kind=kLoop>;
labelloc = t;
tooltip = " ";
3869813376 [label=<<b>convert.10185.1</b><br/>bf16[10,512,11008]{2,1,0}>, shape=rect, tooltip="", style="filled", fontcolor="black", color="#9e9e9e", fillcolor="white"];
}

}
```
graph_file: xla graph dot file


"""


def main():
  parser = ArgumentParser()
  parser.add_argument(
      "subgraph_id",
      type=str,
      help="subgraph id, example: `digraph G{subgraph cluster_3869814064 {}} where 3869814064 is subgraph id`"
  )
  parser.add_argument("graph_file", type=str, help="xla graph graph dot file")
  parser.add_argument(
      "dst_file", type=str, help="output graph dot file that will be writen")
  args = parser.parse_args()
  sub_graph_nodes = set()
  sub_graph_id = args.subgraph_id
  with open(args.dst_file, "w") as fout:
    # with open(args.subgraph_define_file) as f:
    #   lines = f.readlines()
    #   lines = [line for line in lines if line != "}"]
    #   for line in lines:
    #     if not line.split():
    #       continue
    #     first_id = line.split()[0]
    #     if first_id.isdigit():
    #       sub_graph_nodes.add(int(first_id))
    #   fout.writelines(lines)
    fout.write("digraph G {\n")

    subgraph_count = 0
    edge_count = 0
    with open(args.graph_file) as f:
      in_subgraph_context = False
      for line in f:
        if "subgraph" in line:
          subgraph_count += 1
          if sub_graph_id in line:
            in_subgraph_context = True
            print("found subgraph:", sub_graph_id)
        if "}" in line and sub_graph_id in line and in_subgraph_context:
          in_subgraph_context = False
          fout.write("}\n")
          print("subgraph closure:", sub_graph_id)

        if in_subgraph_context:
          if not line.split():
            continue
          first_id = line.split()[0]
          if first_id.isdigit():
            sub_graph_nodes.add(int(first_id))
          fout.write(line)

        if "->" not in line:
          continue
        cols = line.split()
        if cols[0].isdigit() and cols[2].isdigit():
          from_id, to_id = int(cols[0]), int(cols[2])
          if from_id in sub_graph_nodes or to_id in sub_graph_nodes:
            edge_count += 1
            fout.write(line)
      print("found %d subgraphs, and %d edges belong to subgraph:%s" %
            (subgraph_count, edge_count, sub_graph_id))
    fout.write("\n}")


if __name__ == "__main__":
  main()
