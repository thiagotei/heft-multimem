"""Core code to be used for scheduling a task DAG with HEFT"""

from collections import deque, namedtuple
from math import inf, ceil
from heft.gantt import showGanttChart, saveGanttChart
from types import SimpleNamespace
from enum import Enum

import argparse, logging, json, os, sys, csv
import numpy as np, matplotlib.pyplot as plt
import networkx as nx, pydot, pandas as pd, time

logger = logging.getLogger('heft')

class MemKind(Enum):
    FB = 'GPU_FB_MEM' 
    ZC = 'Z_COPY_MEM'
    SY = 'SYSTEM_MEM'

    @staticmethod
    def key(val):
        if val == 'GPU_FB_MEM':
            return MemKind.FB
        elif val == 'Z_COPY_MEM':
            return MemKind.ZC
        elif val == 'SYSTEM_MEM':
            return MemKind.SY
        else:
            raise RuntimeError("[MemKind] {} value not valid!".format(val))
#   #   #

class ProcKind(Enum):
    LOC = 'LOC_PROC'
    TOC = 'TOC_PROC'
    OMP = 'OMP_PROC'

    @staticmethod
    def key(val):
        if val == 'LOC_PROC':
            return ProcKind.LOC
        elif val == 'TOC_PROC':
            return ProcKind.TOC
        elif val == 'OMP_PROC':
            return ProcKind.OMP
        else:
            raise RuntimeError("[ProcKind] {} value not valid!".format(val))
#   #   #

Proc = namedtuple('Proc', 'id kind node socket')
ScheduleEvent = namedtuple('ScheduleEvent', 'task start end proc')
Machine = namedtuple('Machine', 'procs nodes paths interconnect')

"""
Default computation matrix - taken from Topcuoglu 2002 HEFT paper
computation matrix: v x q matrix with v tasks and q PEs
"""
W0 = np.array([
    [14, 16, 9],
    [13, 19, 18],
    [11, 13, 19],
    [13, 8, 17],
    [12, 13, 10],
    [13, 16, 9],
    [7, 15, 11],
    [5, 11, 14],
    [18, 12, 20],
    [21, 7, 16]
])

"""
Default communication matrix - not listed in Topcuoglu 2002 HEFT paper
communication matrix: q x q matrix with q PEs

Note that a communication cost of 0 is used for a given processor to itself
"""
C0 = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

class RankMetric(Enum):
    MEAN = "MEAN"
    WORST = "WORST"
    BEST = "BEST"
    EDP = "EDP"

class OpMode(Enum):
    EFT = "EFT"
    EDP_REL = "EDP RELATIVE"
    EDP_ABS = "EDP ABSOLUTE"
    ENERGY = "ENERGY"

def updateLoggerLevel(l):
    origlevel = logger.getEffectiveLevel()
    logger.setLevel(l)
    for h in logger.handlers:
        h.setLevel(l)
    return origlevel
###

def schedule_dag(dag, linmodel, time_offset=0, rank_metric=RankMetric.MEAN, 
                mappings=None, machine=None, 
                task_cost_def=1, **kwargs):
    """
    Given an application DAG and a set of matrices specifying PE bandwidth and (task, pe) execution times, computes the HEFT schedule
    of that DAG onto that set of PEs 
    """

    op_mode = kwargs.get("op_mode", OpMode.EFT)

    _self = {
        'machine' : machine,
        'root_node': None,
    }
    _self = SimpleNamespace(**_self)

    #nx.nx_pydot.write_dot(dag, './dag_debug.dot')

    # Nodes with no successors cause the any expression to be empty    
    root_node = [node for node in dag.nodes() if not any(True for _ in dag.predecessors(node))]
    assert len(root_node) == 1, f"Expected a single root node, found {len(root_node)}"
    root_node = root_node[0]
    _self.root_node = root_node
    logger.debug(f"[schedule_dag] root_node: {root_node}")

    num_mappings = len(mappings)
    bestmapping = (sys.maxsize, None, None)

    for m_i, m_file in enumerate(mappings):
        # read mapping
        mapping = readMapping(m_file)
        taskstime = tasksTimeCalc(linmodel, mapping)
        # this is just for testing
        #taskstime['realm_copy'] = 10000
        #taskstime['realm_fill'] = 10000
        m_iter = mapping['iteration']

        logger.info(f"====================== Mapping {m_iter} {m_file} {m_i}/{num_mappings} ======================"); 
        logger.debug(""); 
        logger.info(f"Tasks time:\n{json.dumps(taskstime, indent=2, sort_keys=True)}")
        logger.debug("====================== Performing Rank-U Computation ======================"); 
        logger.debug("")
        start = time.time()
        _compute_ranku_mapping(dag, machine, mapping, taskstime, metric=rank_metric, **kwargs)
        delta = time.time() - start
        logger.info(f"*** Ranku computation finished ({delta:.3f} sec)! ***")

        logger.debug(""); 
        logger.debug("====================== Computing EFT for each (task, processor) pair and scheduling in order of decreasing Rank-U ======================"); 
        logger.debug("")
        sorted_nodes = sorted(dag.nodes(), key=lambda node: dag.nodes()[node]['ranku'], reverse=True)

        if sorted_nodes[0] != root_node:
            logger.critical("Root node was not the first node in the sorted list. Must be a zero-cost and zero-weight placeholder node. Rearranging it so it is scheduled first\n")
            idx = sorted_nodes.index(root_node)
            sorted_nodes[idx], sorted_nodes[0] = sorted_nodes[0], sorted_nodes[idx]
        #

        proc_schedules = {}

        _self_mapping = {
            'task_schedules': {},
            'proc_schedules': proc_schedules,
            'time_offset': time_offset,
            'taskstime' : taskstime,
            'totalcomm' : 0.0
        }
        _self_mapping = SimpleNamespace(**_self_mapping)

        logger.debug(f"task_schedules iniatilized here? {_self_mapping.task_schedules}\n" 
                    f"len comp mat : {len(dag.nodes())}")

        for i in dag.nodes():
            _self_mapping.task_schedules[i] = None

        logger.debug(f"[schedule_dag] {len(_self_mapping.task_schedules)} {len(machine.procs)}")

        for p in machine.procs:
            #logger.info(f"p: {p}")
            if p.id not in _self_mapping.proc_schedules:
            #if i not in _self.proc_schedules:
                #logger.info(f"Creating proc schedule {p.id}")
                _self_mapping.proc_schedules[p.id] = []
                #_self.proc_schedules[i] = []

        for proc in _self_mapping.proc_schedules:
            logger.debug(f"proc: {proc}")
            for schedule_event in _self_mapping.proc_schedules[proc]:
                logger.debug(f"Schedule event in proc {proc}: {schedule_event}")
                _self_mapping.task_schedules[schedule_event.task] = schedule_event

        processor_schedules, _, _ = _schedule_dag_mapping(_self_mapping, dag, machine, sorted_nodes, mapping,
                                                        op_mode, rank_metric)

        finaltime = 0.0
        for proc, jobs in processor_schedules.items():
            logger.debug(f"Processor {proc} has the following jobs:")
            logger.debug(f"\t{jobs}")
            lastjobendtime = jobs[-1].end
            lastjobprockind = jobs[-1].proc.kind.value
            if args.pproc:
                logger.info(f"Mapping {m_file} processor {proc} {lastjobprockind} has {len(jobs)} jobs and finished at {lastjobendtime}.")
            if lastjobendtime > finaltime:
                finaltime = lastjobendtime

        logger.info(f"Total comm {_self_mapping.totalcomm} ns or {_self_mapping.totalcomm*1.0e-6:.3f} ms.")
        logger.info(f"Mapping {mapping['iteration']} {m_file} result time {finaltime:.3f} ns or {finaltime*1.0e-6:.3f} ms.")

        if finaltime < bestmapping[0]:
            bestmapping = (finaltime, m_file, processor_schedules)
            logger.info(f"New best mapping {m_file} {finaltime}!")

        if args.showGantt:
            showGanttChart(processor_schedules)

        if args.saveGantt:
            #m = os.path.basename(m_file).split('.')[0] # remove path and extension
            m = str(m_iter)
            gfname = 'gannt_mapping_'+m+'.png'
            saveGanttChart(processor_schedules, machine, gfname, "Mapping "+m)
            logger.info(f"Save Gantt for {m_file} into {gfname}!")
    ##

    return bestmapping
###

def _schedule_dag_mapping(_self, dag, machine, sorted_nodes, mapping, op_mode, rank_metric=RankMetric.MEAN):
    #logger.info(f"\ntask_schedules:\n{_self.task_schedules}")
    #logger.info(f"\nsorted_nodes:\n{sorted_nodes}")
    nodeacct = {}

    for node in sorted_nodes:
        logger.debug(f"[schedule_dag] checking node {node}")
        if _self.task_schedules[node] is not None:
            continue
        minTaskSchedule = ScheduleEvent(node, inf, inf, Proc(-1, None, None, None))

        # If index task launch a subset of procs will be selected
        selprocs = machine.procs

        taskname = dag.nodes[node]['lgtaskname']

        # This tasks must be selected for automap and index_task_launch
        if taskname in mapping['mapping'] and \
            mapping['mapping'][taskname]['index_task_launch']:
            pnt = dag.nodes[node]['lgpnt']
            total = dag.nodes[node]['itltotal']

            # if index task launch is not all nodes, then all task instances of that 
            # index task launch in node 0 for now.
            selmachnode = 0

            #if index task launch, select which nodes this task can run
            if mapping['mapping'][taskname]['all_nodes']:
                # if index task launch all nodes, split homegeneously among all nodes.
                tpernode = math.ceil(total / len(machine.nodes))
                selmachnode = int(pnt / tpernode)
            #

            selprocs = []

            for proc in machine.procs:
                if proc.node == selmachnode:
                    selprocs.append(proc)
                #
            ##
            #logger.info(f"Redefining procs... {node} {taskname} {instname}\n\t{selprocs}")
        #

        if taskname in nodeacct:
            nodeacct[taskname] += 1
        else:
            nodeacct[taskname] = 1
        #

        for proc in selprocs:
        #for proc in procs:
            #logger.info(f"Scheduling {node} {taskname} {instname} {proc}")
            #logger.info(f"[schedule_dag] comp matrix[{dag.nodes[node]['taskinstname']}, {proc.kind.value}] : {_self.computation_matrix.loc[dag.nodes[node]['taskinstname'], proc.kind.value]} {pd.isna(_self.computation_matrix.loc[dag.nodes[node]['taskinstname'], proc.kind.value])}")
            #if mapping['mapping'][dag.nodes[node]['taskinstname']]['processor-kind'] != proc.kind.value:
            # If processor has NaN value in the computation matrix this task should not use it
            #if pd.isna(_self.computation_matrix.loc[instname, proc.kind.value]):
            if taskname in mapping['mapping'] and mapping['mapping'][taskname]['processor-kind'] != proc.kind.value:
                continue

            taskschedule = _compute_eft(_self, dag, node, proc, mapping, machine, taskname)
            #logger.info(f"Scheduling {node} {taskname} {instname} {proc} {taskschedule.end}")
            if (taskschedule.end < minTaskSchedule.end):
                minTaskSchedule = taskschedule
        ##

        _self.task_schedules[node] = minTaskSchedule
        #_self.proc_schedules[minTaskSchedule.proc].append(minTaskSchedule)
        #_self.proc_schedules[minTaskSchedule.proc] = sorted(_self.proc_schedules[minTaskSchedule.proc], key=lambda schedule_event: (schedule_event.end, schedule_event.start))
        _self.proc_schedules[minTaskSchedule.proc.id].append(minTaskSchedule)
        _self.proc_schedules[minTaskSchedule.proc.id] = sorted(_self.proc_schedules[minTaskSchedule.proc.id], key=lambda schedule_event: (schedule_event.end, schedule_event.start))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('\n')
            for proc, jobs in _self.proc_schedules.items():
                logger.debug(f"Processor {proc} has the following jobs:")
                logger.debug(f"\t{jobs}")
            logger.debug('\n')
        for proc in range(len(_self.proc_schedules)):
            for job in range(len(_self.proc_schedules[proc])-1):
                first_job = _self.proc_schedules[proc][job]
                second_job = _self.proc_schedules[proc][job+1]
                assert first_job.end <= second_job.start, \
                f"Jobs on a particular processor must finish before the next can begin, but job {first_job.task} on processor {first_job.proc} ends at {first_job.end} and its successor {second_job.task} starts at {second_job.start}"
    ##

    dict_output = {}
    for proc_num, proc_tasks in _self.proc_schedules.items():
        for idx, task in enumerate(proc_tasks):
            if idx > 0 and (proc_tasks[idx-1].end - proc_tasks[idx-1].start > 0):
                dict_output[task.task] = (proc_num, idx, [proc_tasks[idx-1].task])
            else:
                dict_output[task.task] = (proc_num, idx, [])

    logger.info(f"Nodes processed:\n{json.dumps(nodeacct, indent=2, sort_keys=True)}")

    return _self.proc_schedules, _self.task_schedules, dict_output

def _scale_by_operating_freq(_self, **kwargs):
    if "operating_freqs" not in kwargs:
        logger.debug("No operating frequency argument is present, assuming at max frequency and values are unchanged")
        return
    return #TODO
    #for pe_num, freq in enumerate(kwargs["operating_freqs"]):
        #_self.computation_matrix[:, pe_num] = _self.computation_matrix[:, pe_num] * (1 + compute_DVFS_performance_slowdown(pe_num, freq))

def _compute_ranku_mapping(dag, machine, mapping, taskstime, metric=RankMetric.MEAN, **kwargs):

    """
    Uses a basic BFS approach to traverse upwards through the graph assigning ranku along the way
    """
    #TODO reimplement this by using out_degree calculation.
    #TODO move this to outside of the loop. No need to this every mapping
    terminal_node = [node for node in dag.nodes() if not any(True for _ in dag.successors(node))]
    assert len(terminal_node) == 1, f"Expected a single terminal node, found {len(terminal_node)}"
    logger.debug(f'terminal node {terminal_node[0]} {dag.nodes[terminal_node[0]]}')
    terminal_node = terminal_node[0]

    nx.set_node_attributes(dag, { terminal_node: 1 }, "ranku")
    visit_queue = deque(dag.predecessors(terminal_node))

    logger.debug(f"\tThe ranku for node {terminal_node} is {dag.nodes()[terminal_node]['ranku']}")

    while visit_queue:
        node = visit_queue.pop()
        while _node_can_be_processed(None, dag, node) is not True:
            try:
                node2 = visit_queue.pop()
            except IndexError:
                raise RuntimeError(f"Node {node} cannot be processed, and there are no other nodes in the queue to process instead!")
            visit_queue.appendleft(node)
            node = node2

        logger.debug(f"Assigning ranku for node: {node}")

        if metric == RankMetric.MEAN:
            max_successor_ranku = -1
            nodetaskname = dag.nodes[node]['lgtaskname'] #dep[0]
            succs = dag.successors(node)

            for succnode in succs:
                logger.debug(f"\tLooking at successor node: {succnode}")

                weight = 1
                succtaskname = dag.nodes[succnode]['lgtaskname'] #dep[3]

                if nodetaskname in mapping['mapping'] and \
                   succtaskname in mapping['mapping']:
                    # For each successor, check all the edges.
                    for dep in dag.get_edge_data(node, succnode).values(): #taskdepweights[succnode]:
                        if 'lgrasrc' not in dep or \
                           'lgradst' not in dep or \
                           'weight'  not in dep:
                            continue
                        #
                        ra_src = dep['lgrasrc'] #dep[2]
                        ra_dst = dep['lgradst'] #dep[5]
                        datasize = float(dep['weight']) #float(dep[6])
                        ra_src_mem = mapping['mapping'][nodetaskname]['regions'][str(ra_src)]
                        ra_dst_mem = mapping['mapping'][succtaskname]['regions'][str(ra_dst)]
                        src_mem = _mem_name_fix(MemKind.key(ra_src_mem))
                        dst_mem = _mem_name_fix(MemKind.key(ra_dst_mem))
                        mempath = src_mem + '_to_' + dst_mem

                        avg = machine.paths[mempath]['avg']
                        weight += avg['latency'] + (datasize / avg['bandwidth'])
                #   ##

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"\tThe edge weight from node {node} to node {succnode} is {weight}, and the ranku for node {succnode} is {dag.nodes()[succnode]['ranku']}")
                #

                val = float(weight) + dag.nodes()[succnode]['ranku']
                if val > max_successor_ranku:
                    max_successor_ranku = val
            ##  #

            if max_successor_ranku < 0:
                raise RuntimeError(f"Expected maximum successor ranku to be greater or equal to 0 but was {max_successor_ranku}")

            node_ranku = taskstime.get(nodetaskname, 1) + max_successor_ranku
            nx.set_node_attributes(dag, { node: node_ranku }, "ranku")
            logger.debug(f"\t>>> Ranku of {node}: {dag.nodes[node]['ranku']}")

        else:
            raise RuntimeError(f"Unrecognied Rank-U metric {metric}, unable to compute upward rank")
        #

        visit_queue.extendleft([prednode for prednode in dag.predecessors(node) if prednode not in visit_queue])

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("")
        for node in dag.nodes():
            logger.debug(f"Node: {node}, Rank U: {dag.nodes()[node]['ranku']}")
        ##
    #
###

def _node_can_be_processed(_self, dag, node):
    """
    Validates that a node is able to be processed in Rank U calculations. Namely, that all of its successors have their Rank U values properly assigned
    Otherwise, errors can occur in processing DAGs of the form
    A
    |\
    | B
    |/
    C
    Where C enqueues A and B, A is popped off, and it is unable to be processed because B's Rank U has not been computed
    """
    for succnode in dag.successors(node):
        if 'ranku' not in dag.nodes()[succnode]:
            logger.debug(f"Attempted to compute the Rank U for node {node} but found that it has an unprocessed successor {succnode}. Will try with the next node in the queue")
            return False
    return True

def _mem_name_fix(memname):
    """ Mapping memory names from mapping to machine model.
    """

    name = 'N/A'
    if memname == MemKind.ZC:
        name = 'sys_mem'
    elif memname == MemKind.FB:
        name = 'gpu_fb_mem'
    elif memname == MemKind.SY:
        name = 'sys_mem'
    else:
        raise RuntimeError(f"Memory kind {memname} not found.")
    #
    return name

def _transfer_time(ra_src_proc, ra_src_mem, ra_dst_proc, ra_dst_mem, datasize, machine):
    """ Return transfer time in nanoseconds.
        Linear model numbers per task are in nanoseconds.
    """

    src_mem = _mem_name_fix(ra_src_mem)
    dst_mem = _mem_name_fix(ra_dst_mem)
    transfer_time = 0.0

    # If in the same node
    if ra_src_proc.node == ra_dst_proc.node:
        if ra_src_mem == MemKind.FB and \
           ra_dst_mem == MemKind.FB and \
           ra_src_proc.id == ra_dst_proc.id:
            return 0.0

        elif ra_src_mem == MemKind.SY and \
             ra_dst_mem == MemKind.SY:
            return 0.0

        elif ra_src_mem == MemKind.ZC and \
             ra_dst_mem == MemKind.ZC:
            return 0.0
        #

        # If in the same socket
        if ra_src_proc.socket == ra_dst_proc.socket:
            pathname = 'intra_socket'
        else:
            pathname = 'inter_socket'
        #
    else:
        pathname = 'inter_node'
    #

    mempath = src_mem+'_to_'+dst_mem

    if mempath not in machine.paths:
        raise RuntimeError(f"Unknown {mempath} in machine paths!")
    #
    if pathname not in machine.paths[mempath]:
        raise RuntimeError(f"Unknown {pathname} in machine paths!")
    #

    path = machine.paths[mempath][pathname]

    for p in path:
        # Dual channle pci_to_host and pci_to_dev not in use yet.
        if p.startswith("pci"):
            p = "pci"
        #

        if p not in machine.interconnect:
            raise RuntimeError(f"Interconnect {p} not found in machine!")
        #

        icon = machine.interconnect[p]

        # latency in ns
        lat_p = icon['latency']
        # bw in GB/s
        bw_p = icon['bandwidth']
        # bw in bytes / ns
        bw_p = bw_p
        #transfer_time +=  lat_p*1.0e6 + (datasize/bw_p)
        transfer_time +=  lat_p + (datasize/bw_p)

    logger.debug(transfer_time)
    return transfer_time

def _calc_comm_time(node, nodetname, proc, prednode, predtname, predproc, edges, mapping, machine):
    """
    Calculate communication cost based on the predecessor node and node according to mapping
    of overlapping region arguments.
    The mapping only defines the kind of memory, here needs to pick on physical memory of that selected kind accessible
    by processor selected.
    """
    #print(mapping)
    #logger.info(f"{node} {edges}")
    commcost = 0.0
    commdecisions = {}

    total_datasize = 0

    #if node in taskdepweights:
    if edges is not None and \
       nodetname in mapping['mapping'] and \
       predtname in mapping['mapping']:
        #print("[calc_comm_time]",prednode, node, node in taskdepweights)
        for dep in edges.values():
            if 'lgrasrc' not in dep or \
               'lgradst' not in dep or \
               'weight'  not in dep:
                continue
            #

            ra_src = dep['lgrasrc'] #dep[2]
            ra_dst = dep['lgradst'] #dep[5]
            datasize = float(dep['weight'])#float(dep[6])
            total_datasize += datasize
            ra_src_mem = mapping['mapping'][predtname]['regions'][str(ra_src)]
            ra_dst_mem = mapping['mapping'][nodetname]['regions'][str(ra_dst)]
            curcommcost = _transfer_time(predproc, MemKind.key(ra_src_mem), proc, MemKind.key(ra_dst_mem), datasize, machine)
            commcost += curcommcost
            #logger.info(f"\t{prednode} {predtname} {predproc.id} {predproc.kind.value} {_mem_name_fix(ra_src_mem)} <-> "
            #            f"{_mem_name_fix(ra_dst_mem)} {proc.kind.value} {proc.id} {nodetname} {node}")
            #logger.info(f"\t\t{datasize} bytes {curcommcost} {commcost}")
#                mincostcommra = sys.maxsize
#                bestmem = None
#                for mem in machinemodel[proc].memories:
#                    costcommra = _transfer_time(prednode, ra_src, scheduled[node][ra_src].mem, node, ra_dst, mem)
#                    if mincostcommra > costcommra:
#                        mincostcommra = costcommra
#                        bestmem = mem
#                    #
#                ##
#                commdecisions[ra_dst] = (mincostcommra, bestmem)
#                commcost += mincostcommra
#            ##
#        ##
        #logger.debug(f"Comm cost {commcost} ns")
#    ##
    if commcost > 0.0 and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"All deps {prednode} {predproc.id} {predproc.kind.value} <-> "
                    f"{proc.kind.value} {proc.id} {node} {total_datasize} bytes {commcost}\n")

    return commcost

def _compute_eft(_self, dag, node, proc, mapping, machine, nodetaskname):
    """
    Computes the EFT of a particular node if it were scheduled on a particular processor
    It does this by first looking at all predecessor tasks of a particular node and determining the earliest time a task would be ready for execution (ready_time)
    It then looks at the list of tasks scheduled on this particular processor and determines the earliest time (after ready_time) a given node can be inserted into this processor's queue
    """
    ready_time = _self.time_offset
    logger.debug(f"Computing EFT for node {node} on processor {proc}")

    for prednode in list(dag.predecessors(node)):
        predjob = _self.task_schedules[prednode]
        assert predjob != None, f"Predecessor nodes must be scheduled before their children, but node {node} has an unscheduled predecessor of {prednode}"
        logger.debug(f"\tLooking at predecessor node {prednode} with job {predjob} to determine ready time")
        if False: # _self.communication_matrix[predjob.proc, proc] == 0:
            ready_time_t = predjob.end
        else:
            commtime = _calc_comm_time(node, dag.nodes[node]['lgtaskname'], proc, prednode, dag.nodes[prednode]['lgtaskname'], predjob.proc, dag.get_edge_data(prednode, node), mapping, machine)
            #ready_time_t = predjob.end + dag[predjob.task][node]['weight'] / _self.communication_matrix[predjob.proc, proc] + _self.communication_startup[predjob.proc]
            _self.totalcomm += commtime
            ready_time_t = predjob.end + commtime
        logger.debug(f"\tNode {prednode} can have its data routed to processor {proc} by time {ready_time_t}")
        if ready_time_t > ready_time:
            ready_time = ready_time_t
    logger.debug(f"\tReady time determined to be {ready_time}")

    computation_time = _self.taskstime.get(nodetaskname, 1)
    logger.debug(f"[_compute_eft] computation_time[{node}, {proc.kind.value}]: {computation_time}")
    job_list = _self.proc_schedules[proc.id]
    for idx in range(len(job_list)):
        prev_job = job_list[idx]
        if idx == 0:
            if (prev_job.start - computation_time) - ready_time > 0:
                logger.debug(f"Found an insertion slot before the first job {prev_job} on processor {proc}")
                job_start = ready_time
                min_schedule = ScheduleEvent(node, job_start, job_start+computation_time, proc)
                break
        if idx == len(job_list)-1:
            job_start = max(ready_time, prev_job.end)
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
        next_job = job_list[idx+1]
        #Start of next job - computation time == latest we can start in this window
        #Max(ready_time, previous job's end) == earliest we can start in this window
        #If there's space in there, schedule in it
        logger.debug(f"\tLooking to fit a job of length {computation_time} into a slot of size {next_job.start - max(ready_time, prev_job.end)}")
        if (next_job.start - computation_time) - max(ready_time, prev_job.end) >= 0:
            job_start = max(ready_time, prev_job.end)
            logger.debug(f"\tInsertion is feasible. Inserting job with start time {job_start} and end time {job_start + computation_time} into the time slot [{prev_job.end}, {next_job.start}]")
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
    else:
        #For-else loop: the else executes if the for loop exits without break-ing, which in this case means the number of jobs on this processor are 0
        min_schedule = ScheduleEvent(node, ready_time, ready_time + computation_time, proc)
    logger.debug(f"\tFor node {node} on processor {proc}, the EFT is {min_schedule}")
    return min_schedule    

def tasksTimeCalc(linmodel, mapping):
    """Calculate computation cost for all tasks in a given according to linear model.
    """
    #origlevel = updateLoggerLevel(logging.DEBUG)

    taskstime = {}
    for taskname, tmap in mapping['mapping'].items():
        pkstr = tmap["processor-kind"]

        totaltime = 0
        data_baseline = linmodel["baseline"]["baseline"]

        if pkstr in data_baseline and len(data_baseline[pkstr]) > 0:
            mkstr = next(iter(data_baseline[pkstr].keys()))
            #print("OPA>>>>>>", mkstr," <<<<<",data_baseline[pkstr])

            if taskname in data_baseline[pkstr][mkstr]:
                taskdfmk = next(iter(data_baseline[pkstr][mkstr][taskname].values()))
                totaltime = taskdfmk
                logger.debug("[taskstime] Using baseline {} {} {} time {}".format(taskname, pkstr,mkstr, taskdfmk))
            else:
                logger.error("[taskstime] ERROR! no {} in data_baseline[{}][{}]!".format(taskname, pkstr, mkstr))
                raise RuntimeError()
            #
        else:
            logger.error("[taskstime] ERROR! no {} in data_baseline!".format(pkstr))
            raise RuntimeError()
        #

        if "regions-used" not in tmap:
            raise RuntimeError("[taskstime] expected regions-used in performance data!")
        #

        for regstr, mkstr in tmap["regions-used"].items():
            if (pkstr == ProcKind.LOC.value or pkstr == ProcKind.OMP.value) and \
                    mkstr == MemKind.ZC.value:
                mkstr = MemKind.SY.value
            #

            if taskname in linmodel and \
               regstr in linmodel[taskname] and \
               pkstr in linmodel[taskname][regstr] and \
               mkstr in linmodel[taskname][regstr][pkstr]:
                val = next(iter(linmodel[taskname][regstr][pkstr][mkstr][taskname].values()))
                adj = val - taskdfmk
                totaltime += adj
                logger.debug("[tasktimeestimate] Using specific {} {} {} time {} adj {} total {}".format(taskname, pkstr, mkstr, val, adj, totaltime))
            else:
                logger.debug("[taskstime] No timings for {} {} {} {}".format(taskname,regstr, pkstr, mkstr))
            #
        ##
        logger.debug(f"Final task {taskname} = {totaltime}")

        taskstime[taskname] = totaltime
    ##
    #print(perfdata) 
    #logger.setLevel(origlevel)
    return taskstime
###

def _cleanRealmNeighbors(dag, neigh, funcneigh, nodeprefix='realm'):
    '''Transively remove all neighbours that are realm nodes.
       Their neighbors are added instead.
    '''
    # Find if there are realm_nodes as predecessors or successors of other realm_nodes
    realm_neighs = [p for p in neigh if p.startswith(nodeprefix)]
    while realm_neighs:
        tmp = []
        # Remove the realm nodes after their preds have been added.
        for p in realm_neighs:
            tmp = tmp + list(funcneigh(p))
            neigh.remove(p)
        #
        realm_neighs = [p for p in tmp if p.startswith(nodeprefix)]
        neigh += tmp
    ##
###

def readDagDot(dot_file, save_dag=False):

    logger.info(f"Loading dot file {dot_file}...")

    dag = nx.drawing.nx_pydot.read_dot(dot_file)
    del_realm = True

    logger.debug(f"{dag}")

    # Add single initial and final node
    init_nodes  = [n for n, deg in dag.in_degree() if deg == 0]
    final_nodes = [n for n, deg in dag.out_degree() if deg == 0]

    if len(init_nodes) > 1:
        inode = "_init_"
        if inode not in dag:
            dag.add_node(inode, lgtaskname=inode,lgidxowner="None")
            for k in init_nodes:
                dag.add_edge(inode, k, weight=1)
        else:
            raise RuntimeError(f"{inode} node name collision. find a new one!")
        #
    #

    if len(final_nodes) > 1:
        fnode = "_final_"
        if fnode not in dag:
            dag.add_node(fnode, lgtaskname=fnode, lgidxowner="None")
            for k in final_nodes:
                dag.add_edge(k, fnode, weight=1)
        else:
            raise RuntimeError(f"{fnode} node name collision. find a new one!")
    #   #

    logger.info(f"{dag}")

    # Calculate how many index task launch based on tasks instances with same owner.
    itlowner = {}
    realmnodes = {}
    nodesacct = {}
    for n, attrs in dag.nodes(data=True):
        if n.startswith('realm'):
            tname = n.rsplit('_', 1)[0]
            if del_realm or logger.isEnabledFor(logging.DEBUG):
                if logger.isEnabledFor(logging.DEBUG):
                    preds = list(dag.predecessors(n))
                    succs = list(dag.successors(n))
                    logger.debug(f"{n} In: {dag.in_degree(n)} {preds}  Out: {dag.out_degree(n)} {succs}")

                if del_realm:
                    #_cleanRealmNeighbors(dag, preds, dag.predecessors)
                    #_cleanRealmNeighbors(dag, succs, dag.successors)

                    #realmnodes[n] = (preds, succs)
                    realmnodes[n] = True
                #
            #
        else:
            tname = attrs['lgtaskname']
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{n} {tname} In: {dag.in_degree(n)} Out: {dag.out_degree(n)} Attrs: {attrs}")
            #

            owner = attrs['lgidxowner'] if 'lgidxowner' in attrs else "None"
            if owner in itlowner:
                itlowner[owner] += 1
            else:
                itlowner[owner] = 1
            #

            # Fix the point string to number
            lgpnt = attrs['lgpnt'] if 'lgpnt' in attrs else '\"None\"'
            attrs['lgpnt'] = int(lgpnt.strip('\"()')) if lgpnt != '\"None\"' else 0
        #

        # Clean up some unecessary attributes from nodes
        unnecessary_attrs =['label', 'fontcolor', 'fontsize', 'shape', 'penwidth']
        for a in unnecessary_attrs:
            if a in attrs:
                del attrs[a]
        ##

        if tname not in nodesacct:
            nodesacct[tname] = 0
        #
        nodesacct[tname] += 1
    ##

    logger.info(nodesacct)

    # After collecting in_edges and out_edges for realm nodes,
    # connects predecessors to successors of each realm node and then remove it 
    # until there are no more realm nodes. The weight of the edges must be kept and there
    # may be multiple edges between each pair of nodes.
    if del_realm:
        logger.debug(f"{len(realmnodes.keys())} realmnodes to be removed...")
        for rn in realmnodes.keys():
            in_e = dag.in_edges(rn, data=True, keys=True)
            out_e = dag.out_edges(rn, data=True, keys=True)

            # For each in edge, connect to all out nodes.
            for isrc, idst, ikey, iattrs in in_e:
                for osrc, odst, okey, oattrs in out_e:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Replacing edge {isrc} -> {idst} and {osrc} -> {odst} by {isrc} -> {odst}. {iattrs} {oattrs}")
                    #

                    weight = 0

                    # RealmFill nodes have no weight for now. So, I just pick one of them be used as weight
                    # of the new edge.
                    if 'weight' in iattrs and 'weight' not in oattrs:
                        weight = int(iattrs['weight'])

                    elif 'weight' not in iattrs and 'weight' in oattrs:
                        weight = int(oattrs['weight'])

                    elif 'weight' in iattrs and 'weight' in oattrs and \
                        iattrs['weight'] != oattrs['weight']:
                        if isrc == '_init_':
                            weight = int(oattrs['weight'])
                        else:
                            raise RuntimeError(f"Weights different when replaing edge. They should be equal!")

                    elif 'weight' in iattrs and 'weight' in oattrs:
                        weight = int(iattrs['weight'])
                    #

                    lgrasrc, lgradst = '\"None\"', '\"None\"'
                    if 'lgrasrc' in iattrs and iattrs['lgrasrc'] != '\"None\"':
                        lgrasrc = int(iattrs['lgrasrc'])

                    if 'lgradst' in oattrs and oattrs['lgradst'] != '\"None\"':
                        lgradst = int(oattrs['lgradst'])

                    dag.add_edge(isrc, odst, weight=weight, lgrasrc=lgrasrc, lgradst=lgradst)
        ##  ##  ##

        # By removing nodes, the edges will also be deleted.
        dag.remove_nodes_from(realmnodes)


        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{dag}")

            nodesacct = {}
            for n, attrs in dag.nodes(data=True):
                if n.startswith('realm'):
                    tname = n.rsplit('_', 1)[0]
                else:
                    tname = attrs['lgtaskname']
                #
                if tname not in nodesacct:
                    nodesacct[tname] = 0
                #
                nodesacct[tname] += 1
            ##
            logger.info(nodesacct)
    ##  #

    #TODO erase this
    # Remove realm nodes, 
#    if False: #del_realm:
#        dag.remove_nodes_from(realmnodes.keys())
#        logger.debug(f"{dag}")
#
#        # Replace edges that were removed
#        for _, (preds, succs) in realmnodes.items():
#            for p in preds:
#                for s in succs:
#                    dag.add_edge(p, s)
#                #
#            #
#        #
#        logger.debug(f"{dag}")
#
#        if logger.isEnabledFor(logging.DEBUG):
#            nodesacct = {}
#            for n, attrs in dag.nodes(data=True):
#                if n.startswith('realm'):
#                    tname = n.rsplit('_', 1)[0]
#                else:
#                    tname = attrs['lgtaskname']
#                #
#                if tname not in nodesacct:
#                    nodesacct[tname] = 0
#                #
#                nodesacct[tname] += 1
#            ##
#            logger.info(nodesacct)
#        #
#    #

    # Add the index task launch total points to each node
    for n, attrs in dag.nodes(data=True):
        owner = attrs['lgidxowner']
        attrs['itltotal'] = itlowner[owner]
    #

    if save_dag:
        dag_filename = os.path.basename(dot_file).split('.')[0] # remove path and extension
        dag_dot = './'+dag_filename+'_parsed.dot'
        nx.drawing.nx_pydot.write_dot(dag, dag_dot)
        logger.info(f"*** Saved dag as {dag_dot} ***")
    #

    if False:
        #pos = nx.spring_layout(dag)
        pos = nx.nx_pydot.graphviz_layout(dag, prog='dot')
        nx.draw(dag, pos=pos, with_labels=False)
        dag_pdf = dag_filename+'.pdf'
        plt.savefig(dag_pdf)
        logger.info(f"*** Saved dag as {dag_pdf} ***")
    #

    return dag
###

def readJson(filepath):
    with open(filepath, 'r') as fd:
        buf = json.load(fd)
    return buf
###

def readMapping(mapping_file):
    return readJson(mapping_file)
###

def readMachineModel(mm_file):
    return readJson(mm_file)
###

def readLinearModel(linmodel_file):
    return readJson(linmodel_file)
###

def createProcs(machm_file):
    machmodel = readMachineModel(machm_file)
    numnodes   = machmodel['num_nodes']
    numsocks   = machmodel['num_sockets_per_node']
    numcpusock = machmodel['num_cpus_per_socket']
    numgpusock = machmodel['num_gpus_per_socket']

    nodes = []
    procs = []
    globalid = 0
    for n in range(numnodes):
        nodes.append([])
        for s in range(numsocks):
            for c in range(numcpusock):
                p = Proc(globalid, ProcKind.LOC, n, s)
                procs.append(p)
                nodes[n].append(p)
                globalid += 1

            for g in range(numgpusock):
                p = Proc(globalid, ProcKind.TOC, n, s)
                procs.append(p)
                nodes[n].append(p)
                globalid += 1
            ##
        ##
    ##

    # Calculate avarages that will be used by ranku per path.
    for mempath, loc in machmodel['paths'].items():
        # mempath: mem <-> mem; loc: intra-sock,inter-sock, inter-node
        avgs = {'latency' : 0, 'bandwidth' : 0 }
        for locname, path in loc.items():
            p_avg_bw = 0
            for p in path:
                if p.startswith('pci'):
                    p = 'pci'
                #
                logger.debug(f"{mempath} {locname} {path} lat: {avgs['latency']} {machmodel['interconnect'][p]['latency']} bw: {p_avg_bw}")
                avgs['latency'] += machmodel['interconnect'][p]['latency']
                p_avg_bw += machmodel['interconnect'][p]['bandwidth']
            ##
            avgs['bandwidth'] += p_avg_bw / len(path) 
            logger.debug(f"{mempath} {locname} bw: {avgs['bandwidth']}")
        ##
        # Latency is not averaged in the path, it accumulates
        # Latency avereges among paths.
        avgs['latency']   /= len(loc)
        avgs['latency']    = int(ceil(avgs['latency']))
        # Bandwidth averaged in the path, and then again among paths.
        avgs['bandwidth'] /= len(loc)

        machmodel['paths'][mempath]['avg'] = avgs
    ##

    logger.debug(json.dumps(machmodel, indent=2))

    return Machine(procs, nodes, machmodel['paths'], machmodel['interconnect'])
###

def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for finding HEFT schedules for given DAG task graphs")
    parser.add_argument("-d", "--dag_file", 
                        help="File containing input DAG to be scheduled. Uses default 10 node dag from Topcuoglu 2002 if none given.", 
                        type=str, default="test/canonicalgraph_task_connectivity.csv")
    parser.add_argument("--linmodel", type=str, required=True,
                        help="File containing linear model data for this application.")
    parser.add_argument("--mmodel", type=str, required=True, dest="machine_model_file",
                        help="File containing the machine model.")
    parser.add_argument("--mappings", type=str, nargs='+', required=True, dest="mapping_files", 
                        help="Files containing mapping decisions.")
    parser.add_argument("-l", "--loglevel", 
                        help="The log level to be used in this module. Default: INFO", 
                        type=str, default="INFO", dest="loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--metric",
                        help="Specify which metric to use when performing upward rank calculation",
                        type=RankMetric, default=RankMetric.MEAN, dest="rank_metric", choices=list(RankMetric))
    parser.add_argument("--showDAG", 
                        help="Switch used to enable display of the incoming task DAG", 
                        dest="showDAG", action="store_true")
    parser.add_argument("--saveDAG", 
                        help="Save the incoming task DAG", 
                        dest="saveDAG", action="store_true")
    parser.add_argument("--showGantt", 
                        help="Switch used to enable display of the final scheduled Gantt chart", 
                        dest="showGantt", action="store_true")
    parser.add_argument("--saveGantt", 
                        help="Switch used to save the scheduled Gantt chart for each mapping", 
                        action="store_true")
    parser.add_argument("--pproc",
                        help="Show times per processor.",
                        action="store_true")
    parser.add_argument("--parseDAGonly",
                        help="For debugging, run up to dag parsing and stops.",
                        action="store_true")

    return parser
###

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    start = time.time()
    logger.setLevel(logging.getLevelName(args.loglevel))
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.getLevelName(args.loglevel))
    if logger.isEnabledFor(logging.DEBUG):
        consolehandler.setFormatter(logging.Formatter("%(levelname)8s %(asctime)s : %(name)s %(funcName)s : %(message)s"))
    else:
        consolehandler.setFormatter(logging.Formatter("%(levelname)8s %(asctime)s : %(name)s : %(message)s"))
    #
    logger.addHandler(consolehandler)

    start_inpread = time.time()

    # processors data structure based on mach model
    machine = createProcs(args.machine_model_file)

    logger.info(f"*** Machine model created! ***")

    linmodel = readLinearModel(args.linmodel)

    logger.info(f"*** Linear model parsed! ***")

    dag = readDagDot(args.dag_file, save_dag=args.saveDAG)

    delta_inp = time.time() - start_inpread
    logger.info(f"*** Task graph created ({delta_inp:.3f} sec)! ***")

    if args.parseDAGonly:
        logger.warning("Stopping after parsing the task graph!!!")
        sys.exit(0)
    #

    start_maps = time.time()
    besttime, best_mfile, processor_schedules = schedule_dag(dag, linmodel, rank_metric=args.rank_metric, 
                                            mappings=args.mapping_files, machine=machine)
    delta_maps = time.time() - start_maps

    totaltime = 0.0
    for proc, jobs in processor_schedules.items():
        logger.debug(f"Processor {proc} has the following jobs:")
        logger.debug(f"\t{jobs}")
        lastjobendtime = jobs[-1].end
        lastjobprockind = jobs[-1].proc.kind.value
        logger.debug(f"Mapping {best_mfile} processor {proc} {lastjobprockind} has {len(jobs)} jobs and finished at {lastjobendtime}.")
        if lastjobendtime > totaltime:
            totaltime = lastjobendtime

    logger.info(f"Best mapping {best_mfile} final result time {totaltime:.3f} ns or {totaltime*1.0e-6:.3f} ms.")
    num_mappings = len(args.mapping_files)
    end = time.time()
    delta = end - start
    logger.info(f"Simulator processed {num_mappings} mappings in {delta_maps:.3f} sec (avg: {delta_maps/num_mappings:.3f} sec/mapping), input read {delta_inp:.3f} sec, total {delta:.3f} sec.")

    if args.showGantt:
        showGanttChart(processor_schedules)
