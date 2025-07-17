# coding=utf-8
from __future__ import unicode_literals, absolute_import
import json
from argparse import ArgumentParser
from collections import defaultdict


def main():
    parser = ArgumentParser()
    parser.add_argument('req_json', type=str, help="reorder request file")
    parser.add_argument('res_json', type=str, help="reorder result file")
    parser.add_argument('trace_dst_file', type=str,
                        help="write chrome trace file path")

    args = parser.parse_args()
    with open(args.req_json) as fin:
        req_json = json.load(fin)
    with open(args.res_json) as fin:
        res_json = json.load(fin)
    uuid2node = {}
    nodes = req_json["nodes"]
    for node in nodes:
        uuid2node[node["uuid"]] = node
    # cat,dur,ts,
    edge_map = {}
    edges = req_json["edges"]
    for edge in edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if uuid2node[from_id]["typename"] != "compute" and \
                uuid2node[to_id]["typename"] != "compute":  # comm start node
            edge_map[from_id] = edge
    traceEvents = []
    chrome_trace_json_format = {
        "distributedInfo": {
            "rank": 0
        },

    }
    stackFrames = {}
    if "finalized_node_start_end_time" in res_json:
        finalized_node_start_end_time = res_json["finalized_node_start_end_time"]
        # prepare comm edge
        for node_start_end in finalized_node_start_end_time:
            node_uuid = node_start_end["node_uuid"]
            if "_" in node_uuid:  # TODO: edge,
                continue
            node = uuid2node[node_uuid]
            if node["typename"] == "compute":
                traceEvents.append({
                    "id": node_uuid,
                    "cat": "kernel",
                    "dur": node_start_end["end_time"]-node_start_end["start_time"],
                    "ts": node_start_end["start_time"],
                    "pid": 0,
                    "tid": 0,
                    "name": node["name"],
                    "ph": "X",
                    "args": {
                        "cost": node["cost"],
                        "uuid": node_uuid

                    }
                })
            else:
                name = "ncclKernel_"+node["name"]
                if node_uuid not in edge_map:
                    continue
                cost = edge_map[node_uuid]["cost"]
                traceEvents.append({
                    "id": node_uuid,
                    "cat": "kernel",
                    "dur": cost,
                    "ts": node_start_end["start_time"],
                    "pid": 0,
                    "tid": 1,
                    "name": name,
                    "ph": "X",
                    "args": {
                        "cost": cost,
                        "uuid": node_uuid

                    }
                })
    elif "nodes" in res_json:
        # ortools format
        # prepare communication edge
        node_endat = {}
        edge_map = {}
        node2deps = defaultdict(list)
        for edge in req_json["edges"]:
            node2deps[edge["to"]].append(edge["from"])
            if edge["from"] not in stackFrames:
                stackFrames[edge["from"]] = {
                    "name": edge["from"],
                    "id": edge["from"]
                }
        for node in res_json["nodes"]:
            node_before_solve = uuid2node[node["uuid"]]
            # overwrite
            node["opcode"] = node_before_solve["opcode"]
            uuid2node[node["uuid"]] = node
            if node["typename"] == "communication":
                head = node["name"].split("=")[0].strip()
                node_endat[head] = node
                if "done." in head:  # it's done node
                    start_node_key = head.replace("done.", "start.")
                    start_node = node_endat[start_node_key]
                    edge_map[start_node_key] = {
                        "from": start_node_key,
                        "to": head,
                        "cost": node["endTime"]-start_node["startTime"]
                    }
        for node in res_json["nodes"]:
            uuid = node["uuid"]
            if node["typename"] == "compute":
                traceEvents.append({
                    "id": uuid,
                    "cat": "kernel",
                    "dur": node["endTime"]-node["startTime"],
                    "ts": node["startTime"],
                    "pid": 0,
                    "tid": 0,
                    "name": node["name"],
                    "ph": "X",
                    "args": {
                        "cost": node["endTime"]-node["startTime"],
                        "uuid": uuid,
                        "opcode": node["opcode"],
                        "obj": {
                            "id_ref": node2deps[uuid]
                        },
                        "deps": node2deps[uuid]
                    }
                })
            else:
                name = "ncclKernel_"+node["name"]
                head = node["name"].split("=")[0].strip()
                if head in edge_map:
                    cost = edge_map[head]["cost"]
                else:
                    cost = node["endTime"]-node["startTime"]
                traceEvents.append({
                    "id": uuid,
                    "cat": "kernel",
                    "dur": cost,
                    "ts": node["startTime"],
                    "pid": 0,
                    "tid": 1,
                    "name": node["name"],
                    "ph": "X",
                    "args": {
                        "cost": cost,
                        "uuid": uuid,
                        "opcode": node["opcode"]

                    }
                })
        for edge in req_json["edges"]:

            from_id = edge["from"]
            to_id = edge["to"]
            start_node = uuid2node[from_id]
            to_node = uuid2node[to_id]
            edge_id = "%s-%s" % (from_id, to_id)
            # Async Events,use same id; b (nestable start), n (nestable instant), e (nestable end)
            traceEvents.append({
                "id": from_id,
                "cat": "flow",
                "ts": start_node["startTime"],
                "pid": 1,
                "tid": 3,
                "name": start_node["name"],
                "ph": "s",
                "args": {
                    "from_id": from_id,
                    "to_id": to_id,
                }
            })

            traceEvents.append({
                "id": from_id,
                "cat": "flow",
                "ts": to_node["endTime"],
                "pid": 1,
                "tid": 3,
                "name": to_node["name"],
                "ph": "f",
                "bp": "e",
                "args": {
                    "from_id": from_id,
                    "to_id": to_id,
                }
            })
            break

    chrome_trace_json_format["traceEvents"] = traceEvents
    with open(args.trace_dst_file, "w") as fout:
        json.dump(chrome_trace_json_format, fout, indent=4)


if __name__ == '__main__':
    main()
