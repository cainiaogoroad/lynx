from ortools.sat.python import cp_model
import json
import random
import numpy as np
random.seed(19260817)
np.random.seed(19260817)

dag = {
    "nodes": [
        {
            "id": "JOB0",
            "name": "JOB0",
            "dependencies": [],
            "cost": [10, 1, 10]
        },
        {
            "id": "JOB1",
            "name": "JOB1",
            "dependencies": ["JOB0"],
            "cost": [10, 10, 10]
        },
        {
            "id": "JOB2",
            "name": "JOB2",
            "dependencies": ["JOB1"],
            "cost": [10, 1, 10]
        },
        {
            "id": "JOB3",
            "name": "JOB3",
            "dependencies": ["JOB0", "JOB1"],
            "cost": [10, 10, 10]
        },
        {
            "id": "JOB4",
            "name": "JOB4",
            "dependencies": ["JOB0", "JOB1"],
            "cost": [1, 10, 1]
        },
        {
            "id": "JOB5",
            "name": "JOB5",
            "dependencies": ["JOB0", "JOB1"],
            "cost": [10, 1, 10]
        }
    ]
}

def solve_dag(dags, solve_file='chrome_trace.json'):

    model = cp_model.CpModel()

    total_time = sum(sum(node['cost']) for node in dags['nodes'])
    horizon = total_time

    jobs = {}
    for node in dags['nodes']:
        job_id = node['id']
        dependencies = node['dependencies']
        cost = node['cost']
        read_time, tc_execute_time, ele_execute_time, write_time = cost

        read_start = model.NewIntVar(0, horizon, f"{job_id}_read_start")
        read_end = model.NewIntVar(0, horizon, f"{job_id}_read_end")
        model.Add(read_end == read_start + read_time)
        read_interval = model.NewIntervalVar(read_start, read_time, read_end, f"{job_id}_read")

        tc_execute_start = model.NewIntVar(0, horizon, f"{job_id}_tc_execute_start")
        tc_execute_end = model.NewIntVar(0, horizon, f"{job_id}_tc_execute_end")
        model.Add(tc_execute_end == tc_execute_start + tc_execute_time)
        tc_execute_interval = model.NewIntervalVar(tc_execute_start, tc_execute_time, tc_execute_end, f"{job_id}_tc_execute")

        ele_execute_start = model.NewIntVar(0, horizon, f"{job_id}_ele_execute_start")
        ele_execute_end = model.NewIntVar(0, horizon, f"{job_id}_ele_execute_end")
        model.Add(ele_execute_end == ele_execute_start + ele_execute_time)
        ele_execute_interval = model.NewIntervalVar(ele_execute_start, ele_execute_time, ele_execute_end, f"{job_id}_ele_execute")

        write_start = model.NewIntVar(0, horizon, f"{job_id}_write_start")
        write_end = model.NewIntVar(0, horizon, f"{job_id}_write_end")
        model.Add(write_end == write_start + write_time)
        write_interval = model.NewIntervalVar(write_start, write_time, write_end, f"{job_id}_write")

        model.Add(tc_execute_start >= read_end)
        model.Add(ele_execute_start >= read_end)
        model.Add(write_start >= ele_execute_end)
        model.Add(write_start >= tc_execute_end)

        jobs[job_id] = {
            'read': {'start': read_start, 'end': read_end, 'interval': read_interval},
            'tc_execute': {'start': tc_execute_start, 'end': tc_execute_end, 'interval': tc_execute_interval},
            'ele_execute': {'start': ele_execute_start, 'end': ele_execute_end, 'interval': ele_execute_interval},
            'write': {'start': write_start, 'end': write_end, 'interval': write_interval},
            'dependencies': dependencies
        }

    read_intervals = [jobs[job_id]['read']['interval'] for job_id in jobs]
    write_intervals = [jobs[job_id]['write']['interval'] for job_id in jobs]
    demands = [1] * (len(read_intervals)+len(write_intervals))
    model.AddCumulative(read_intervals+write_intervals, demands, 1)

    tc_execute_intervals = [jobs[job_id]['tc_execute']['interval'] for job_id in jobs]
    demands = [1] * len(tc_execute_intervals)
    model.AddCumulative(tc_execute_intervals, demands, 1)

    ele_execute_intervals = [jobs[job_id]['ele_execute']['interval'] for job_id in jobs]
    demands = [1] * len(ele_execute_intervals)
    model.AddCumulative(ele_execute_intervals, demands, 1)

    for job_id in jobs:
        job = jobs[job_id]
        for dep_id in job['dependencies']:
            dep_job = jobs[dep_id]
            model.Add(job['read']['start'] >= dep_job['write']['end'])

    write_ends = [jobs[job_id]['write']['end'] for job_id in jobs]
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, write_ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"optimal result(makespan): {solver.ObjectiveValue()}")
        trace_events = []
        job_ids = [node['id'] for node in dags['nodes']]
        jobs_order = {job_id: idx for idx, job_id in enumerate(job_ids)}
        thread_ids = {
            "read": 0,
            "tc_execute": 1,
            "ele_execute": 2,
            "write": 3
        }
        for node in dags['nodes']:
            job_id = node['id']
            job = jobs[job_id]
            cost = node['cost']
            read_dur, tc_execute_dur, ele_execute_dur, write_dur = cost

            read_start = solver.Value(job['read']['start'])
            trace_events.append({
                'name': f"{job_id} Read",
                'cat': 'READ',
                'ph': 'X',
                'ts': read_start,
                'dur': read_dur,
                'pid': 0,
                'tid': thread_ids['read'],
                'args': {"dep_jobs": job['dependencies']}
            })

            tc_execute_start = solver.Value(job['tc_execute']['start'])
            trace_events.append({
                'name': f"{job_id} TC Execute",
                'cat': 'TC_EXECUTE',
                'ph': 'X',
                'ts': tc_execute_start,
                'dur': tc_execute_dur,
                'pid': 0,
                'tid': thread_ids['tc_execute'],
                'args': {"dep_jobs": job['dependencies']}
            })

            ele_execute_start = solver.Value(job['ele_execute']['start'])
            trace_events.append({
                'name': f"{job_id} CUDA Execute",
                'cat': 'CUDA_EXECUTE',
                'ph': 'X',
                'ts': ele_execute_start,
                'dur': ele_execute_dur,
                'pid': 0,
                'tid': thread_ids['ele_execute'],
                'args': {"dep_jobs": job['dependencies']}
            })

            write_start = solver.Value(job['write']['start'])
            trace_events.append({
                'name': f"{job_id} Write",
                'cat': 'WRITE',
                'ph': 'X',
                'ts': write_start,
                'dur': write_dur,
                'pid': 0,
                'tid': thread_ids['write'],
                'args': {"dep_jobs": job['dependencies']}
            })

        with open(solve_file, 'w') as f:
            json.dump({'traceEvents': trace_events}, f, indent=2)
        print(f"Trace dumped to file {solve_file}")
    else:
        print("No feasible solution")

def make_test_data(num_jobs=100):

    nodes = []
    for i in range(num_jobs):
        nodes.append({
            "id": f"JOB{i}",
            "name": f"JOB{i}",
            "dependencies": [],
            "cost": [random.randint(1, 100), 
                     random.randint(1, 100),
                     random.randint(1, 100),
                     random.randint(1, 100)]
        })

    for i in range(num_jobs):
        this_node = nodes[i]
        max_nodes = 3
        for j in range(i):
            if random.random() < 0.5 and (len(this_node['dependencies']) < max_nodes):
                this_node['dependencies'].append(nodes[j]['id'])
    return {"nodes":nodes}


make_test_data(100)
solve_dag(make_test_data(100), 'chrome_trace_random.json')