"""Core code to be used for scheduling a task DAG with HEFT"""

from collections import deque, namedtuple
from math import inf, ceil
from heft.gantt import showGanttChart, saveGanttChart
from types import SimpleNamespace
from enum import Enum

import argparse
import logging
import json, os
import sys, csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import time

logger = logging.getLogger('heft')

class MemKind(Enum):
    FB = 'GPU_FB_MEM' 
    ZC = 'Z_COPY_MEM'
    SY = 'SYSTEM_MEM'
#

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
            raise RuntimeError("[Processor] {} value not valid!".format(val))
        #
    #
#

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

def schedule_dag(dag, linmodel, proc_schedules_inp=None, 
                time_offset=0, relabel_nodes=True, rank_metric=RankMetric.MEAN, 
                mappings=None, taskdepweights=None, machine=None, task_inst_names=None, 
                task_cost_def=1, **kwargs):
    """
    Given an application DAG and a set of matrices specifying PE bandwidth and (task, pe) execution times, computes the HEFT schedule
    of that DAG onto that set of PEs 
    """

    op_mode = kwargs.get("op_mode", OpMode.EFT)

    _self = {
        'machine' : machine,
        'taskdepweights' : taskdepweights,
        'root_node': None,
    }
    _self = SimpleNamespace(**_self)

    nx.nx_pydot.write_dot(dag, './dag_debug.dot')

    # Nodes with no successors cause the any expression to be empty    
    root_node = [node for node in dag.nodes() if not any(True for _ in dag.predecessors(node))]
    assert len(root_node) == 1, f"Expected a single root node, found {len(root_node)}"
    root_node = root_node[0]
    _self.root_node = root_node
    logger.debug(f"[schedule_dag] root_node: {root_node} {dag.nodes[root_node]['taskinstname']}")

    num_mappings = len(mappings)
    bestmapping = (sys.maxsize, None, None)

    for m_i, m_file in enumerate(mappings):
        #receive mapping
        mapping = readMapping(m_file)
        taskstime = tasksTimeCalc(linmodel, mapping)
        m_iter = mapping['iteration']

        logger.info(f"====================== Mapping {m_iter} {m_file} {m_i}/{num_mappings} ======================"); 
        logger.debug(""); 
        logger.info(f"Tasks time:\n{json.dumps(taskstime, indent=2, sort_keys=True)}")
        logger.debug("====================== Performing Rank-U Computation ======================"); 
        logger.debug("")
        _compute_ranku_mapping(dag, machine, mapping, taskstime, taskdepweights, metric=rank_metric, **kwargs)

        logger.debug(""); 
        logger.debug("====================== Computing EFT for each (task, processor) pair and scheduling in order of decreasing Rank-U ======================"); 
        logger.debug("")
        sorted_nodes = sorted(dag.nodes(), key=lambda node: dag.nodes()[node]['ranku'], reverse=True)

        if sorted_nodes[0] != root_node:
            logger.critical("Root node was not the first node in the sorted list. Must be a zero-cost and zero-weight placeholder node. Rearranging it so it is scheduled first\n")
            idx = sorted_nodes.index(root_node)
            sorted_nodes[idx], sorted_nodes[0] = sorted_nodes[0], sorted_nodes[idx]
        #

        if proc_schedules_inp == None:
            proc_schedules = {}
        else:
            proc_schedules = copy.deepcopy(proc_schedules_inp)
        #

        _self_mapping = {
            'task_schedules': {},
            'proc_schedules': proc_schedules,
            'numExistingJobs': 0,
            'time_offset': time_offset,
            'taskstime' : taskstime,
            'totalcomm' : 0.0
        }
        _self_mapping = SimpleNamespace(**_self_mapping)

        for proc in proc_schedules:
            logger.debug(f"Update num existing jobs")
            _self_mapping.numExistingJobs = _self_mapping.numExistingJobs + len(proc_schedules[proc])
        ##

        if relabel_nodes:
            logger.debug(f"Relabel nodes {relabel_nodes} {_self_mapping.numExistingJobs}")
            dag = nx.relabel_nodes(dag, dict(map(lambda node: (node, node+_self_mapping.numExistingJobs), list(dag.nodes()))))
        else:
            #Negates any offsets that would have been needed had the jobs been relabeled
            _self_mapping.numExistingJobs = 0
        #

        logger.debug(f"task_schedules iniatilized here? {_self_mapping.task_schedules}\n" 
                    f"numExistingJobs {_self_mapping.numExistingJobs} "
                    f"len comp mat : {len(dag.nodes())}")

        for i in range(_self_mapping.numExistingJobs + len(dag.nodes())):
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
                                                        op_mode, rank_metric, task_inst_names, taskdepweights)

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

def _schedule_dag_mapping(_self, dag, machine, sorted_nodes, mapping, op_mode, rank_metric=RankMetric.MEAN, task_inst_names=None, taskdepweights=None):
    #logger.info(f"\ntask_schedules:\n{_self.task_schedules}")
    #logger.info(f"\nsorted_nodes:\n{sorted_nodes}")
    nodeacct = {}

    for node in sorted_nodes:
        logger.debug(f"[schedule_dag] checking node {node} {dag.nodes[node]['taskinstname']}")
        if _self.task_schedules[node] is not None:
            continue
        minTaskSchedule = ScheduleEvent(node, inf, inf, Proc(-1, None, None, None))
        minEDP = inf
        if op_mode == OpMode.EDP_ABS:
            assert "power_dict" in kwargs, "In order to perform EDP-based processor assignment, a power_dict is required"
            taskschedules = []
            minScheduleStart = inf

            for proc in machine.procs:
                taskschedule = _compute_eft(_self, dag, node, proc)
                edp_t = ((taskschedule.end - taskschedule.start)**2) * kwargs["power_dict"][node][proc]
                if (edp_t < minEDP):
                    minEDP = edp_t
                    minTaskSchedule = taskschedule
                elif (edp_t == minEDP and taskschedule.end < minTaskSchedule.end):
                    minTaskSchedule = taskschedule

        elif op_mode == OpMode.EDP_REL:
            assert "power_dict" in kwargs, "In order to perform EDP-based processor assignment, a power_dict is required"
            taskschedules = []
            minScheduleStart = inf

            for proc in machine.procs:
                taskschedules.append(_compute_eft(_self, dag, node, proc))
                if taskschedules[proc].start < minScheduleStart:
                    minScheduleStart = taskschedules[proc].start

            for taskschedule in taskschedules:
                # Use the makespan relative to the earliest potential assignment to encourage load balancing
                edp_t = ((taskschedule.end - minScheduleStart)**2) * kwargs["power_dict"][node][taskschedule.proc]
                if (edp_t < minEDP):
                    minEDP = edp_t
                    minTaskSchedule = taskschedule
                elif (edp_t == minEDP and taskschedule.end < minTaskSchedule.end):
                    minTaskSchedule = taskschedule

        elif op_mode == OpMode.ENERGY:
            assert False, "Feature not implemented"
            assert "power_dict" in kwargs, "In order to perform Energy-based processor assignment, a power_dict is required"

        else:
            # If index task launch a subset of procs will be selected
            selprocs = machine.procs

            instname = dag.nodes[node]['taskinstname']
            taskname = 'N/A'
            for t, insts in task_inst_names.items():
                if instname in insts:
                    taskname = t
                    break
                #
            ##

            # This tasks must be selected for automap and index_task_launch
            if taskname in mapping['mapping'] and \
                mapping['mapping'][taskname]['index_task_launch']:
                pnt = dag.nodes[node]['itlpnt']
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

                taskschedule = _compute_eft(_self, dag, node, proc, taskdepweights, mapping, machine, taskname)
                #logger.info(f"Scheduling {node} {taskname} {instname} {proc} {taskschedule.end}")
                if (taskschedule.end < minTaskSchedule.end):
                    minTaskSchedule = taskschedule

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
    
    dict_output = {}
    for proc_num, proc_tasks in _self.proc_schedules.items():
        for idx, task in enumerate(proc_tasks):
            if idx > 0 and (proc_tasks[idx-1].end - proc_tasks[idx-1].start > 0):
                dict_output[task.task] = (proc_num, idx, [proc_tasks[idx-1].task])
            else:
                dict_output[task.task] = (proc_num, idx, [])

    logger.info(f"Node acct:\n{json.dumps(nodeacct, indent=2, sort_keys=True)}")

    return _self.proc_schedules, _self.task_schedules, dict_output

def _scale_by_operating_freq(_self, **kwargs):
    if "operating_freqs" not in kwargs:
        logger.debug("No operating frequency argument is present, assuming at max frequency and values are unchanged")
        return
    return #TODO
    #for pe_num, freq in enumerate(kwargs["operating_freqs"]):
        #_self.computation_matrix[:, pe_num] = _self.computation_matrix[:, pe_num] * (1 + compute_DVFS_performance_slowdown(pe_num, freq))

def _compute_ranku_mapping(dag, machine, mapping, taskstime, taskdepweights, metric=RankMetric.MEAN, **kwargs):

    """
    Uses a basic BFS approach to traverse upwards through the graph assigning ranku along the way
    """
    terminal_node = [node for node in dag.nodes() if not any(True for _ in dag.successors(node))]
    assert len(terminal_node) == 1, f"Expected a single terminal node, found {len(terminal_node)}"
    logger.debug(f'terminal node {terminal_node[0]} {dag.nodes[terminal_node[0]]}')
    terminal_node = terminal_node[0]
    node_names = nx.get_node_attributes(dag, "taskinstname")

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
            nodetaskname = 'N/A'
            for succnode in dag.successors(node):
                logger.debug(f"\tLooking at successor node: {succnode}")

                weight = 1
                if node_names[succnode] in taskdepweights:
                    for dep in taskdepweights[node_names[succnode]]:
                        if dep[1] == node_names[node]:
                            pdtaskname = dep[0]
                            nodetaskname = dep[3]
                            ra_src = dep[2]
                            ra_dst = dep[5]
                            datasize = float(dep[6])
                            ra_src_mem = mapping['mapping'][pdtaskname]['regions'][str(ra_src)]
                            ra_dst_mem = mapping['mapping'][nodetaskname]['regions'][str(ra_dst)]
                            src_mem = _mem_name_fix(ra_src_mem)
                            dst_mem = _mem_name_fix(ra_dst_mem)
                            mempath = src_mem + '_to_' + dst_mem
 
                            avg = machine.paths[mempath]['avg']
                            weight += avg['latency'] + (datasize / avg['bandwidth'])
                        #
                    ##
                #
                logger.debug(f"\tThe edge weight from node {node} to node {succnode} is {weight}, and the ranku for node {succnode} is {dag.nodes()[succnode]['ranku']}")

                val = float(weight) + dag.nodes()[succnode]['ranku']
                if val > max_successor_ranku:
                    max_successor_ranku = val
                #
            ##

            assert max_successor_ranku >= 0, f"Expected maximum successor ranku to be greater or equal to 0 but was {max_successor_ranku}"

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
            logger.debug(f"Attempted to compute the Rank U for node {node} but found that it has an unprocessed successor {dag.nodes()[succnode]}. Will try with the next node in the queue")
            return False
    return True

def _mem_name_fix(memname):
    """ Mapping memory names from mapping to machine model.
    """

    #print(memname)
    name = 'N/A'
    if memname == MemKind.ZC.value:
        name = 'sys_mem'
    elif memname == MemKind.FB.value:
        name = 'gpu_fb_mem'
    elif memname == MemKind.SY.value:
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

def _calc_comm_time(node, nodeattrs, proc, prednode, prednodeattrs, predproc, taskdepweights, mapping, machine):
    """
    Calculate communication cost based on the predecessor node and node according to mapping
    of overlapping region arguments.
    The mapping only defines the kind of memory, here needs to pick on physical memory of that selected kind accessible
    by processor selected.
    """
    #print(mapping)
    commcost = 0.0
    commdecisions = {}

    nodename = nodeattrs["taskinstname"]
    prednodename = prednodeattrs["taskinstname"]
    total_datasize = 0

    if nodename in taskdepweights:
        #print("[calc_comm_time]",prednode, prednodename, node, nodename, nodename in taskdepweights)
        for dep in taskdepweights[nodename]:
            #print("       ",dep )
            pdtaskname = dep[0]
            nodetaskname   = dep[3]
            ra_src = dep[2]
            ra_dst = dep[5]
            datasize = float(dep[6])
            if dep[1] == prednodename:
                total_datasize += datasize
                ra_src_mem = mapping['mapping'][pdtaskname]['regions'][str(ra_src)]
                ra_dst_mem = mapping['mapping'][nodetaskname]['regions'][str(ra_dst)]
                #print(pdtaskname,ra_src,ra_src_mem, nodetaskname, ra_dst, ra_dst_mem, datasize, type(datasize))
                #commcost += datasize*(1/10)
                #commcost += _transfer_time(ra_src_proc, ra_src_mem, ra_dst_proc, ra_dst_mem, datasize, machinemodel)
                curcommcost = _transfer_time(predproc, ra_src_mem, proc, ra_dst_mem, datasize, machine)
                commcost += curcommcost
                logger.info(f"[_calc_comm_time]\t{prednodename} {pdtaskname} {predproc.id} {predproc.kind.value} {_mem_name_fix(ra_src_mem)} <-> "
                            f"{_mem_name_fix(ra_dst_mem)} {proc.kind.value} {proc.id} {nodetaskname} {nodename}")
                logger.info(f"\t\t{datasize} bytes {curcommcost} {commcost}")
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
    if commcost > 0.0:
        logger.info(f"[_calc_comm_time] All deps {prednodename} {predproc.id} {predproc.kind.value} <-> "
                    f"{proc.kind.value} {proc.id} {nodename} {total_datasize} bytes {commcost}\n")

    return commcost

def _compute_eft(_self, dag, node, proc, taskdepweights, mapping, machine, nodetaskname):
    """
    Computes the EFT of a particular node if it were scheduled on a particular processor
    It does this by first looking at all predecessor tasks of a particular node and determining the earliest time a task would be ready for execution (ready_time)
    It then looks at the list of tasks scheduled on this particular processor and determines the earliest time (after ready_time) a given node can be inserted into this processor's queue
    """
    #print(type(dag.nodes[node]), dag.nodes[node])
    ready_time = _self.time_offset
    logger.debug(f"Computing EFT for node {node} on processor {proc}")
    #for ra of node.ras:
        # for mem of proc:
            # if mem == mapping[node.name][ra]:

    node_name = dag.nodes[node]['taskinstname']

    for prednode in list(dag.predecessors(node)):
        predjob = _self.task_schedules[prednode]
        prednode_name = dag.nodes[prednode]['taskinstname']
        assert predjob != None, f"Predecessor nodes must be scheduled before their children, but node {node} {node_name} has an unscheduled predecessor of {prednode} {prednode_name}"
        logger.debug(f"\tLooking at predecessor node {prednode} with job {predjob} to determine ready time")
        if False: # _self.communication_matrix[predjob.proc, proc] == 0:
            ready_time_t = predjob.end
        else:
            commtime = _calc_comm_time(node, dag.nodes[node], proc, prednode, dag.nodes[prednode], predjob.proc, taskdepweights, mapping, machine)
            #ready_time_t = predjob.end + dag[predjob.task][node]['weight'] / _self.communication_matrix[predjob.proc, proc] + _self.communication_startup[predjob.proc]
            _self.totalcomm += commtime
            ready_time_t = predjob.end + commtime
        logger.debug(f"\tNode {prednode} can have its data routed to processor {proc} by time {ready_time_t}")
        if ready_time_t > ready_time:
            ready_time = ready_time_t
    logger.debug(f"\tReady time determined to be {ready_time}")

    #computation_time = _self.computation_matrix[node-_self.numExistingJobs, proc.kind.value]
    computation_time = _self.taskstime.get(nodetaskname, 1) #_self.computation_matrix.loc[node_name, proc.kind.value]
    logger.debug(f"[_compute_eft] computation_time[{node_name}, {proc.kind.value}]: {computation_time}")
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

def readCsvToNumpyMatrix(csv_file):
    """
    Given an input file consisting of a comma separated list of numeric values with a single header row and header column, 
    this function reads that data into a numpy matrix and strips the top row and leftmost column
    """
    with open(csv_file) as fd:
        logger.debug(f"Reading the contents of {csv_file} into a matrix")
        contents = fd.read()
        contentsList = contents.split('\n')
        contentsList = list(map(lambda line: line.split(','), contentsList))
        contentsList = list(filter(lambda arr: arr != [''], contentsList))
        
        matrix = np.array(contentsList)
        # Matrix has id for each node, but we need the actula node names.
        nodenames = list(matrix[0, 1:])
        matrix = np.delete(matrix, 0, 0) # delete the first row (entry 0 along axis 0)
        matrix = np.delete(matrix, 0, 1) # delete the first column (entry 0 along axis 1)
        matrix = matrix.astype(float)
        logger.debug(f"Nodenames:\n{nodenames}")
        logger.debug(f"After deleting the first row and column of input data, we are left with this matrix:\n{matrix}")
        return matrix, nodenames

def readDagMatrix(dag_file, itlfile, show_dag=False):
    """
    Given an input file consisting of a connectivity matrix, reads and parses it into a networkx Directional Graph (DiGraph)
    """
    matrix, nodesnames = readCsvToNumpyMatrix(dag_file)

    dag = nx.DiGraph(matrix)
    dag.remove_edges_from(
        # Remove all edges with weight of 0 since we have no placeholder for "this edge doesn't exist" in the input file
        [edge for edge in dag.edges() if dag.get_edge_data(*edge)['weight'] == '0.0']
    )

    with open(itlfile, 'r') as fd:
        csvfile = csv.reader(fd)
        itld = {}
        itlowner = {}

        for line in csvfile:
            ind = line[0]
            info = line[1:]
            itld[ind] = info
            # Count how many points in the index domain by adding tasks with same owner.
            # Used to know which node place the task.
            owner = info[1]
            if owner in itlowner:
                itlowner[owner] += 1
            else:
                itlowner[owner] = 1
        ##

        attrs = {}
        for n in dag.nodes:
            if n in itld:
                itl = True
                pnt = itld[2]
                owner = itld[1]
                tltpnt = itlowner[owner]
            else:
                itl = False
                pnt = -1
                tltpnt = -1
            #

            attrs[n] = {"taskinstname" : nodesnames[n],
                        "itl" : itl,
                        "itlpnt" : pnt,
                        "itltotal" : tltpnt}

    nx.set_node_attributes(dag, attrs)
    #print("Blah:\n",dag.nodes[0]["taskinstname"])
    #print("Blah:\n", attrs)

    #print(list(dag.nodes))
    if show_dag:
        nx.draw(dag, pos=nx.nx_pydot.graphviz_layout(dag, prog='dot'), with_labels=True)
        plt.show()

    return dag

def readMultiDagMatrix(dag_file, show_dag=False):
    """
    Given an input file consisting of a connectivity matrix, reads and parses it into a networkx Directional Graph (DiGraph)
    """
    #matrix = readCsvToNumpyMatrix(dag_file)
    with open(dag_file) as fd:
        logger.debug(f"Reading the contents of {dag_file} into a matrix")
        contents = fd.read()
        contentsList = contents.split('\n')
        contentsList = list(map(lambda line: line.split(','), contentsList))
        contentsList = list(filter(lambda arr: arr != [''], contentsList))
        logger.debug(f"Content list:\n{contentsList}")

        newDag = nx.MultiDiGraph()

        #Ignore first line and first column
        for line, lval in enumerate(contentsList[1:]):
            for col, cval in enumerate(lval[1:]):
                wvals = cval.split('-')
                for widx, w in enumerate(wvals):
                    wv = float(w)
                    if wv != 0.0:
                        NEwDag.add_edge(line, col, key=widx, rsize=wv)
                ##
            ##
        ##
        logger.debug(f"After deleting the first row and column of input data, we are left with this matrix:\n{newDag}")

        
        #matrix = np.array(contentsList)
        #matrix = np.delete(matrix, 0, 0) # delete the first row (entry 0 along axis 0)
        #matrix = np.delete(matrix, 0, 1) # delete the first column (entry 0 along axis 1)
        #matrix = matrix.astype(float)
        #logger.debug(f"After deleting the first row and column of input data, we are left with this matrix:\n{matrix}")

        #dag = nx.MultiDiGraph(matrix)
        #dag.remove_edges_from(
            # Remove all edges with weight of 0 since we have no placeholder for "this edge doesn't exist" in the input file
        #    [edge for edge in dag.edges() if dag.get_edge_data(*edge)['weight'] == '0.0']
        #)

        if show_dag:
            nx.draw(dag, pos=nx.nx_pydot.graphviz_layout(dag, prog='dot'), with_labels=True)
            plt.show()

    return newDag

def readMapping(mapping_file):

    with open(mapping_file, 'r') as m_open_file:
        mapping = json.load(m_open_file)

    return mapping

def readMachineModel(mm_file):
    with open(mm_file, 'r') as mm_fd:
        machmodel = json.load(mm_fd)

    return machmodel

def readTaskDepWeigths(taskdep_file):
    """
    Read a file with a dependence between tasks per region argument in bytes.
    """
    with open(taskdep_file, 'r') as fd:
        contents = fd.read()
        contentsList = contents.split('\n')
        contentsList = list(map(lambda line: line.split(','), contentsList))
        contentsList = list(filter(lambda arr: arr != [''], contentsList))

        # a dict per destination task instance
        taskdepwgts = {}
        #print(contentsList)

        for r in contentsList:
            assert len(r) == 7,f"Expecting 7 fieds on taskdep file, but got {len(r)}"
            if r[4] not in taskdepwgts:
                taskdepwgts[r[4]] = []
            #

            taskdepwgts[r[4]].append(r)
        #

    return taskdepwgts


def readTaskNames(tasknames_file):
    with open(tasknames_file, 'r') as fd:
        contents = fd.read()
        contentsList = contents.split('\n')
        contentsList = list(map(lambda line: line.split(','), contentsList))
        contentsList = list(filter(lambda arr: arr != [''], contentsList))

        nodestasknames = {}

        for line in contentsList:
            if line[0] not in nodestasknames:
                nodestasknames[line[0]] = []
            #
            for node in line[1:]:
                nodestasknames[line[0]].append(node)
            ##
        ##

    return nodestasknames

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

def readLinearModel(linmodel_file):
    # Read the linear model file
    perfdata = None
    with open(linmodel_file, 'r') as f:
        data = f.read()
        perfdata = json.loads(data)
    #
    return perfdata
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
                        help="File containing mapping decisions.")
    parser.add_argument("--taskdepra", type=str, required=True, dest="task_dep_weights",
                        help="File containing the machine model.")
    parser.add_argument("--tasknames", type=str, required=True, dest="taskinstancesnames",
                        help="File containing the task names each node represent." )
    parser.add_argument("--indexl", type=str, required=True, dest="indexl",
                        help="File containing the index task launch information for each dag node." )
    parser.add_argument("-l", "--loglevel", 
                        help="The log level to be used in this module. Default: INFO", 
                        type=str, default="INFO", dest="loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--metric",
                        help="Specify which metric to use when performing upward rank calculation",
                        type=RankMetric, default=RankMetric.MEAN, dest="rank_metric", choices=list(RankMetric))
    parser.add_argument("--showDAG", 
                        help="Switch used to enable display of the incoming task DAG", 
                        dest="showDAG", action="store_true")
    parser.add_argument("--showGantt", 
                        help="Switch used to enable display of the final scheduled Gantt chart", 
                        dest="showGantt", action="store_true")
    parser.add_argument("--saveGantt", 
                        help="Switch used to save the scheduled Gantt chart for each mapping", 
                        action="store_true")
    parser.add_argument("--pproc",
                        help="Show times per processor.",
                        action="store_true")

    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    start = time.time()
    logger.setLevel(logging.getLevelName(args.loglevel))
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.getLevelName(args.loglevel))
    consolehandler.setFormatter(logging.Formatter("%(levelname)8s : %(name)16s : %(message)s"))
    logger.addHandler(consolehandler)

    # receive taskdepcomm
    taskdepcomm = readTaskDepWeigths(args.task_dep_weights)
    # processors data structure based on mach model
    machine = createProcs(args.machine_model_file)
    # nodes to task names
    tasknames = readTaskNames(args.taskinstancesnames)
    #logger.debug(f"tasknames:\n{tasknames}")

    linmodel = readLinearModel(args.linmodel)

    dag = readDagMatrix(args.dag_file, args.indexl, args.showDAG)
    # For now, the dag do not weight per region argument between task instances.
    #dag = readMultiDagMatrix(args.dag_file, args.showDAG)

    besttime, best_mfile, processor_schedules = schedule_dag(dag, linmodel, rank_metric=args.rank_metric, 
                                            mappings=args.mapping_files, taskdepweights=taskdepcomm, machine=machine, 
                                            task_inst_names=tasknames)

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
    logger.info(f"Simulator processed {num_mappings} mappings in {delta:.2f} sec (avg: {delta/num_mappings:.2f} sec/mapping).")

    if args.showGantt:
        showGanttChart(processor_schedules)
