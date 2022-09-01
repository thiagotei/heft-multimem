"""Core code to be used for scheduling a task DAG with HEFT"""

from collections import deque, namedtuple
from math import inf
from heft.gantt import showGanttChart
from types import SimpleNamespace
from enum import Enum

import argparse
import logging
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

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

"""
Default communication startup cost vector
"""
L0 = np.array([0, 0, 0])

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

def schedule_dag(dag, computation_matrix=W0, communication_matrix=C0, communication_startup=L0, proc_schedules=None, time_offset=0, relabel_nodes=True, rank_metric=RankMetric.MEAN, mem_mem_matrix=None, mem_pe_matrix=None, mapping=None, taskdepweights=None, machinemodel=None, task_inst_names=None, procs=None, pkindscol=None, **kwargs):
    """
    Given an application DAG and a set of matrices specifying PE bandwidth and (task, pe) execution times, computes the HEFT schedule
    of that DAG onto that set of PEs 
    """
    if proc_schedules == None:
        proc_schedules = {}

    _self = {
        'computation_matrix': computation_matrix,
        'communication_matrix': communication_matrix,
        'communication_startup': communication_startup,
        'mem_mem_matrix' : mem_mem_matrix,
        'mem_pe_matrix' : mem_pe_matrix,
        'machinemodel' : machinemodel,
        'taskdepweights' : taskdepweights,
        'task_schedules': {},
        'proc_schedules': proc_schedules,
        'numExistingJobs': 0,
        'time_offset': time_offset,
        'root_node': None
    }
    _self = SimpleNamespace(**_self)

    for proc in proc_schedules:
        logger.info(f"Update num existing jobs")
        _self.numExistingJobs = _self.numExistingJobs + len(proc_schedules[proc])

    if relabel_nodes:
        logger.info(f"Relabel nodes {relabel_nodes} {_self.numExistingJobs}")
        dag = nx.relabel_nodes(dag, dict(map(lambda node: (node, node+_self.numExistingJobs), list(dag.nodes()))))
    else:
        #Negates any offsets that would have been needed had the jobs been relabeled
        _self.numExistingJobs = 0

    logger.info(f"task_schedules iniatilized here? {_self.task_schedules}\n numExistingJobs {_self.numExistingJobs} len comp mat : {len(_self.computation_matrix)}")
    for i in range(_self.numExistingJobs + len(_self.computation_matrix)):
        _self.task_schedules[i] = None

    logger.info(f"[schedule_dag] {len(_self.task_schedules)} {len(computation_matrix)}")

    #for i in range(len(_self.communication_matrix)):
    for p in procs:
        #logger.info(f"p: {p}")
        if p.id not in _self.proc_schedules:
        #if i not in _self.proc_schedules:
            #logger.info(f"Creating proc schedule {p.id}")
            _self.proc_schedules[p.id] = []
            #_self.proc_schedules[i] = []

    for proc in proc_schedules:
        logger.info(f"proc: {proc}")
        for schedule_event in proc_schedules[proc]:
            logger.info(f"Schedule event in proc {proc}: {schedule_event}")
            _self.task_schedules[schedule_event.task] = schedule_event

    # Nodes with no successors cause the any expression to be empty    
    root_node = [node for node in dag.nodes() if not any(True for _ in dag.predecessors(node))]
    assert len(root_node) == 1, f"Expected a single root node, found {len(root_node)}"
    root_node = root_node[0]
    _self.root_node = root_node
    logger.info(f"[schedule_dag] root_node: {root_node} {dag.nodes[root_node]['taskinstname']}")

    logger.debug(""); logger.debug("====================== Performing Rank-U Computation ======================\n"); logger.debug("")
    _compute_ranku(_self, dag, metric=rank_metric, **kwargs)

    logger.debug(""); logger.debug("====================== Computing EFT for each (task, processor) pair and scheduling in order of decreasing Rank-U ======================"); logger.debug("")
    sorted_nodes = sorted(dag.nodes(), key=lambda node: dag.nodes()[node]['ranku'], reverse=True)
    if sorted_nodes[0] != root_node:
        logger.critical("Root node was not the first node in the sorted list. Must be a zero-cost and zero-weight placeholder node. Rearranging it so it is scheduled first\n")
        idx = sorted_nodes.index(root_node)
        sorted_nodes[idx], sorted_nodes[0] = sorted_nodes[0], sorted_nodes[idx]

    #nx.nx_pydot.write_dot(dag, './dag_debug.dot')

    #logger.info(f"\ntask_schedules:\n{_self.task_schedules}")
    #logger.info(f"\nsorted_nodes:\n{sorted_nodes}")
    for node in sorted_nodes:
        logger.debug(f"[schedule_dag] checking node {node} {dag.nodes[node]['taskinstname']}")
        if _self.task_schedules[node] is not None:
            continue
        minTaskSchedule = ScheduleEvent(node, inf, inf, Proc(-1, None, None, None))
        minEDP = inf
        op_mode = kwargs.get("op_mode", OpMode.EFT)
        if op_mode == OpMode.EDP_ABS:
            assert "power_dict" in kwargs, "In order to perform EDP-based processor assignment, a power_dict is required"
            taskschedules = []
            minScheduleStart = inf

            #for proc in range(len(communication_matrix)):
            for proc in procs:
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

            #for proc in range(len(communication_matrix)):
            for proc in procs:
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
            #for proc in range(len(communication_matrix)):
            for proc in procs:
                #logger.info(f"[schedule_dag] comp matrix[{dag.nodes[node]['taskinstname']}, {proc.kind.value}] : {_self.computation_matrix.loc[dag.nodes[node]['taskinstname'], proc.kind.value]} {pd.isna(_self.computation_matrix.loc[dag.nodes[node]['taskinstname'], proc.kind.value])}")
                #if mapping['mapping'][dag.nodes[node]['taskinstname']]['processor-kind'] != proc.kind.value:
                # If processor has NaN value in the computation matrix this task should not use it
                if pd.isna(_self.computation_matrix.loc[dag.nodes[node]['taskinstname'], proc.kind.value]):
                    continue
                logger.debug(f" What is a node? {type(node)} {node}")
                taskschedule = _compute_eft(_self, dag, node, proc, taskdepweights, mapping, machinemodel, task_inst_names, pkindscol)
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

    return _self.proc_schedules, _self.task_schedules, dict_output

def _scale_by_operating_freq(_self, **kwargs):
    if "operating_freqs" not in kwargs:
        logger.debug("No operating frequency argument is present, assuming at max frequency and values are unchanged")
        return
    return #TODO
    #for pe_num, freq in enumerate(kwargs["operating_freqs"]):
        #_self.computation_matrix[:, pe_num] = _self.computation_matrix[:, pe_num] * (1 + compute_DVFS_performance_slowdown(pe_num, freq))

def _compute_ranku(_self, dag, metric=RankMetric.MEAN, **kwargs):
    """
    Uses a basic BFS approach to traverse upwards through the graph assigning ranku along the way
    """
    terminal_node = [node for node in dag.nodes() if not any(True for _ in dag.successors(node))]
    assert len(terminal_node) == 1, f"Expected a single terminal node, found {len(terminal_node)}"
    logger.info(f'terminal node {terminal_node[0]} {dag.nodes[terminal_node[0]]}')
    terminal_node = terminal_node[0]
    node_attrs = nx.get_node_attributes(dag, "taskinstname")

    #TODO: Should this be configurable?
    #avgCommunicationCost = np.mean(_self.communication_matrix[np.where(_self.communication_matrix > 0)])
    diagonal_mask = np.ones(_self.communication_matrix.shape, dtype=bool)
    np.fill_diagonal(diagonal_mask, 0)
    avgCommunicationCost = np.mean(_self.communication_matrix[diagonal_mask]) + np.mean(_self.communication_startup)
    for edge in dag.edges():
        logger.debug(f"Assigning {edge}'s average weight based on average communication cost. {float(dag.get_edge_data(*edge)['weight'])} => {float(dag.get_edge_data(*edge)['weight']) / avgCommunicationCost}")
        nx.set_edge_attributes(dag, { edge: float(dag.get_edge_data(*edge)['weight']) / avgCommunicationCost }, 'avgweight')

    # Utilize a masked array so that np.mean, etc, calculations ignore the entries that are inf
    #comp_matrix_masked = np.ma.masked_where(_self.computation_matrix == inf, _self.computation_matrix)
    comp_matrix_masked = _self.computation_matrix #.isin([np.nan, np.inf, -np.inf]).any(1)
    #logger.info(f"comp_matrix_masked {comp_matrix_masked}")
    logger.info(f"comp_matrix_masked mean {comp_matrix_masked.loc[node_attrs[terminal_node]].mean()}")

    nx.set_node_attributes(dag, { terminal_node: comp_matrix_masked.loc[node_attrs[terminal_node]].mean() }, "ranku")
    visit_queue = deque(dag.predecessors(terminal_node))

    while visit_queue:
        node = visit_queue.pop()
        while _node_can_be_processed(_self, dag, node) is not True:
            try:
                node2 = visit_queue.pop()
            except IndexError:
                raise RuntimeError(f"Node {node} cannot be processed, and there are no other nodes in the queue to process instead!")
            visit_queue.appendleft(node)
            node = node2

        logger.debug(f"Assigning ranku for node: {node}")
        if metric == RankMetric.MEAN:
            max_successor_ranku = -1
            for succnode in dag.successors(node):
                logger.debug(f"\tLooking at successor node: {succnode}")
                logger.debug(f"\tThe edge weight from node {node} to node {succnode} is {dag[node][succnode]['avgweight']}, and the ranku for node {node} is {dag.nodes()[succnode]['ranku']}")
                val = float(dag[node][succnode]['avgweight']) + dag.nodes()[succnode]['ranku']
                if val > max_successor_ranku:
                    max_successor_ranku = val
            assert max_successor_ranku >= 0, f"Expected maximum successor ranku to be greater or equal to 0 but was {max_successor_ranku}"
            node_ranku = comp_matrix_masked.loc[node_attrs[node]].mean(skipna=True) + max_successor_ranku
            nx.set_node_attributes(dag, { node: comp_matrix_masked.loc[node_attrs[node]].mean(skipna=True) + max_successor_ranku }, "ranku")
            logger.debug(f"\t>>> Ranku of {node}: {dag.nodes[node]['ranku']}")

        elif metric == RankMetric.WORST:
            max_successor_ranku = -1
            max_node_idx = np.where(comp_matrix_masked[node-_self.numExistingJobs] == max(comp_matrix_masked[node-_self.numExistingJobs]))[0][0]
            logger.debug(f"\tNode {node} has maximum computation cost of {comp_matrix_masked[node-_self.numExistingJobs][max_node_idx]} on processor {max_node_idx}")
            for succnode in dag.successors(node):
                logger.debug(f"\tLooking at successor node: {succnode}")
                max_succ_idx = np.where(comp_matrix_masked[succnode-_self.numExistingJobs] == max(comp_matrix_masked[succnode-_self.numExistingJobs]))[0][0]
                logger.debug(f"\tNode {succnode} has maximum computation cost of {comp_matrix_masked[succnode-_self.numExistingJobs][max_succ_idx]} on processor {max_succ_idx}")
                val = _self.communication_matrix[max_node_idx, max_succ_idx] + dag.nodes()[succnode]['ranku']
                if val > max_successor_ranku:
                    max_successor_ranku = val
            assert max_successor_ranku >= 0, f"Expected maximum successor ranku to be greater or equal to 0 but was {max_successor_ranku}"
            nx.set_node_attributes(dag, { node: comp_matrix_masked[node-_self.numExistingJobs, max_node_idx] + max_successor_ranku}, "ranku")
        
        elif metric == RankMetric.BEST:
            min_successor_ranku = inf
            min_node_idx = np.where(comp_matrix_masked[node-_self.numExistingJobs] == min(comp_matrix_masked[node-_self.numExistingJobs]))[0][0]
            logger.debug(f"\tNode {node} has minimum computation cost on processor {min_node_idx}")
            for succnode in dag.successors(node):
                logger.debug(f"\tLooking at successor node: {succnode}")
                min_succ_idx = np.where(comp_matrix_masked[succnode-_self.numExistingJobs] == min(comp_matrix_masked[succnode-_self.numExistingJobs]))[0][0]
                logger.debug(f"\tThis successor node has minimum computation cost on processor {min_succ_idx}")
                val = _self.communication_matrix[min_node_idx, min_succ_idx] + dag.nodes()[succnode]['ranku']
                if val < min_successor_ranku:
                    min_successor_ranku = val
            assert min_successor_ranku >= 0, f"Expected minimum successor ranku to be greater or equal to 0 but was {min_successor_ranku}"
            nx.set_node_attributes(dag, { node: comp_matrix_masked[node-_self.numExistingJobs, min_node_idx] + min_successor_ranku}, "ranku")
        
        elif metric == RankMetric.EDP:
            assert "power_dict" in kwargs, "In order to perform EDP-based Rank Method, a power_dict is required"
            power_dict = kwargs.get("power_dict", np.array([[]]))
            power_dict_masked = np.ma.masked_where(power_dict[node] == inf, power_dict[node])
            max_successor_ranku = -1
            for succnode in dag.successors(node):
                logger.debug(f"\tLooking at successor node: {succnode}")
                logger.debug(f"\tThe edge weight from node {node} to node {succnode} is {dag[node][succnode]['avgweight']}, and the ranku for node {node} is {dag.nodes()[succnode]['ranku']}")
                val = float(dag[node][succnode]['avgweight']) + dag.nodes()[succnode]['ranku']
                if val > max_successor_ranku:
                    max_successor_ranku = val
            assert max_successor_ranku >= 0, f"Expected maximum successor ranku to be greater or equal to 0 but was {max_successor_ranku}"
            avg_edp = np.mean(comp_matrix_masked[node-_self.numExistingJobs])**2 * np.mean(power_dict_masked)
            nx.set_node_attributes(dag, { node: avg_edp + max_successor_ranku }, "ranku")
        
        else:
            raise RuntimeError(f"Unrecognied Rank-U metric {metric}, unable to compute upward rank")

        visit_queue.extendleft([prednode for prednode in dag.predecessors(node) if prednode not in visit_queue])
    
    logger.debug("")
    for node in dag.nodes():
        logger.debug(f"Node: {node}, Rank U: {dag.nodes()[node]['ranku']}")

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

def _transfer_time(ra_src_proc, ra_src_mem, ra_dst_proc, ra_dst_mem, datasize, machinemodel):
    """ Return transfer time in nanoseconds.
        Linear model numbers per task are in nanoseconds.
    """

    src_mem = _mem_name_fix(ra_src_mem)
    dst_mem = _mem_name_fix(ra_dst_mem)
    transfer_time = 0.0

    # If in the same node
        # If in the same socket

    pathname = src_mem+'_to_'+dst_mem

    if pathname not in machinemodel['paths']:
        raise RuntimeError(f"Unknown {pathname} in machine model paths!")
    #

    path = machinemodel['paths'][pathname]['intra_socket']
        # not in the same socket
        #else
        #

    # not in the same node
    #else
    #

    for p in path:
        # Dual channle pci_to_host and pci_to_dev not in use yet.
        if p.startswith("pci"):
            p = "pci"
        #

        if p not in machinemodel['interconnect']:
            raise RuntimeError(f"Interconnect {p} not found in machine model!")
        #

        icon = machinemodel['interconnect'][p]

        # latency in ms
        lat_p = icon['latency']
        # bw in GB/s
        bw_p = icon['bandwidth']
        # bw in bytes / ns
        bw_p = bw_p
        transfer_time +=  lat_p*1.0e6 + (datasize/bw_p)

    logger.debug(transfer_time)
    return transfer_time

def _calc_comm_time(node, nodeattrs, prednode, prednodeattrs, proc, taskdepweights, mapping, machinemodel, task_inst_names):
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
                ra_src_mem = mapping['mapping'][pdtaskname]['regions'][str(ra_src)]
                ra_dst_mem = mapping['mapping'][nodetaskname]['regions'][str(ra_dst)]
                #print(pdtaskname,ra_src,ra_src_mem, nodetaskname, ra_dst, ra_dst_mem, datasize, type(datasize))
                #commcost += datasize*(1/10)
                #commcost += _transfer_time(ra_src_proc, ra_src_mem, ra_dst_proc, ra_dst_mem, datasize, machinemodel)
                commcost += _transfer_time(None, ra_src_mem, None, ra_dst_mem, datasize, machinemodel)
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
    return commcost

def _compute_eft(_self, dag, node, proc, taskdepweights, mapping, machinemodel, task_inst_names, pkindscol):
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
            commtime = _calc_comm_time(node, dag.nodes[node], prednode, dag.nodes[prednode], proc, taskdepweights, mapping, machinemodel, task_inst_names)
            #ready_time_t = predjob.end + dag[predjob.task][node]['weight'] / _self.communication_matrix[predjob.proc, proc] + _self.communication_startup[predjob.proc]
            ready_time_t = predjob.end + commtime
        logger.debug(f"\tNode {prednode} can have its data routed to processor {proc} by time {ready_time_t}")
        if ready_time_t > ready_time:
            ready_time = ready_time_t
    logger.debug(f"\tReady time determined to be {ready_time}")

    #computation_time = _self.computation_matrix[node-_self.numExistingJobs, proc.kind.value]
    computation_time = _self.computation_matrix.loc[node_name, proc.kind.value]
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

def readCsvToPandas(csv_file):
    """
    Given an input file consisting of a comma separated list of numeric values with a single header row and header column, 
    this function reads that data into a numpy matrix and strips the top row and leftmost column
    """
    df = pd.read_csv(csv_file)
    #print(df.dtypes)
    logger.debug(f"Reading the contents of {csv_file} into a dataframe:\n{df}")
    return df

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

def readCsvToDict(csv_file):
    """
    Given an input file consisting of a comma separated list of numeric values with a single header row and header column, 
    this function reads that data into a dictionary with keys that are node numbers and values that are the CSV lists
    """
    with open(csv_file) as fd:
        matrix = readCsvToNumpyMatrix(csv_file)
        
        outputDict = {}
        for row_num, row in enumerate(matrix):
            outputDict[row_num] = row
        return outputDict

def readDagMatrix(dag_file, show_dag=False):
    """
    Given an input file consisting of a connectivity matrix, reads and parses it into a networkx Directional Graph (DiGraph)
    """
    matrix, nodesnames = readCsvToNumpyMatrix(dag_file)

    dag = nx.DiGraph(matrix)
    dag.remove_edges_from(
        # Remove all edges with weight of 0 since we have no placeholder for "this edge doesn't exist" in the input file
        [edge for edge in dag.edges() if dag.get_edge_data(*edge)['weight'] == '0.0']
    )

    attrs = {}
    for n in dag.nodes:
        attrs[n] = {"taskinstname" : nodesnames[n]}

    nx.set_node_attributes(dag, attrs)
    #print("Blah:\n",dag.nodes[0]["taskinstname"])
    #print("Blah:\n", attrs)

    #print(list(dag.nodes))
    if show_dag:
        nx.draw(dag, pos=nx.nx_pydot.graphviz_layout(dag, prog='dot'), with_labels=True)
        plt.show()

    return dag

def readMultiCsvToNumpyMatrix(csv_file):
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
        matrix = np.delete(matrix, 0, 0) # delete the first row (entry 0 along axis 0)
        matrix = np.delete(matrix, 0, 1) # delete the first column (entry 0 along axis 1)
        matrix = matrix.astype(float)
        logger.debug(f"After deleting the first row and column of input data, we are left with this matrix:\n{matrix}")
        return matrix

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
            for node in line[1:]:
                nodestasknames[line[0]] = node
            ##
        ##

    return nodestasknames

def createProcs(machmodel):
    numnodes   = machmodel['num_nodes']
    numsocks   = machmodel['num_sockets_per_node']
    numcpusock = machmodel['num_cpus_per_socket']
    numgpusock = machmodel['num_gpus_per_socket']

    procs = []
    globalid = 0
    for n in range(numnodes):
        for s in range(numsocks):
            for c in range(numcpusock):
                procs.append(Proc(globalid, ProcKind.LOC, n, s))
                globalid += 1

            for g in range(numgpusock):
                procs.append(Proc(globalid, ProcKind.TOC, n, s))
                globalid += 1

    return procs
#

def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for finding HEFT schedules for given DAG task graphs")
    parser.add_argument("-d", "--dag_file", 
                        help="File containing input DAG to be scheduled. Uses default 10 node dag from Topcuoglu 2002 if none given.", 
                        type=str, default="test/canonicalgraph_task_connectivity.csv")
    parser.add_argument("-p", "--pe_connectivity_file", 
                        help="File containing connectivity/bandwidth information about PEs. Uses a default 3x3 matrix from Topcuoglu 2002 if none given. If communication startup costs (L) are needed, a \"Startup\" row can be used as the last CSV row", 
                        type=str, default="test/canonicalgraph_resource_BW.csv")
    parser.add_argument("-t", "--task_execution_file", 
                        help="File containing execution times of each task on each particular PE. Uses a default 10x3 matrix from Topcuoglu 2002 if none given.", 
                        type=str, default="test/canonicalgraph_task_exe_time.csv")
#    parser.add_argument("--memconn", type=str, required=True,
#                        help="File containing connectivity/bandwidth information about Memory elements.")
#    parser.add_argument("--mempe", type=str, required=True,
#                        help="File containing connectivity between memory and processors.")
    parser.add_argument("--mmodel", type=str, required=True, dest="machine_model_file",
                        help="File containing the machine model.")
    parser.add_argument("--mapping", type=str, required=True, dest="mapping_file", 
                        help="File containing mapping decisions.")
    parser.add_argument("--taskdepra", type=str, required=True, dest="task_dep_weights",
                        help="File containing the machine model.")
    parser.add_argument("--tasknames", type=str, required=True, dest="taskinstancesnames",
                        help="File containing the task names each node represent." )
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
    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    logger.setLevel(logging.getLevelName(args.loglevel))
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.getLevelName(args.loglevel))
    consolehandler.setFormatter(logging.Formatter("%(levelname)8s : %(name)16s : %(message)s"))
    logger.addHandler(consolehandler)

    #receive mapping
    mapping = readMapping(args.mapping_file)
    # receive taskdepcomm
    taskdepcomm = readTaskDepWeigths(args.task_dep_weights)
    # machine mode desc
    machmodel = readMachineModel(args.machine_model_file)
    # processors data structure based on mach model
    procs = createProcs(machmodel)
    # nodes to task names
    tasknames = readTaskNames(args.taskinstancesnames)
    communication_matrix,_ = readCsvToNumpyMatrix(args.pe_connectivity_file)
    #computation_matrix, pkindnames = readCsvToNumpyMatrix(args.task_execution_file)
    #"nda)
    computation_matrix = readCsvToPandas(args.task_execution_file)
    computation_matrix.set_index('T', inplace=True)
    pkindnames = [] # TODO remove this
    #mem_pe_matrix = readCsvToNumpyMatrix(args.mempe)
    #mem_mem_matrix = readCsvToNumpyMatrix(args.memconn)

    dag = readDagMatrix(args.dag_file, args.showDAG) 
    # For now, the dag do not weight per region argument between task instances.
    #dag = readMultiDagMatrix(args.dag_file, args.showDAG)
    
    if (communication_matrix.shape[0] != communication_matrix.shape[1]):
        assert communication_matrix.shape[0]-1 == communication_matrix.shape[1], "If the communication_matrix CSV is non-square, there must only be a single additional row specifying the communication startup costs of each PE"
        logger.debug("Non-square communication matrix parsed. Stripping off the last row as communication startup costs");
        communication_startup = communication_matrix[-1, :]
        communication_matrix = communication_matrix[0:-1, :]
    else:
        communication_startup = np.zeros(communication_matrix.shape[0])

    logger.debug(f"Comput matrix index?\n{computation_matrix}")

    # Check computation_matrix and dag have same nodes
    set_dag = set(x for i,x in dag.nodes(data="taskinstname"))
    #logger.info(f"set_dag: {set_dag}")
    #set_comp_mat =  set(computation_matrix['T'].tolist())
    set_comp_mat =  set(computation_matrix.index.values.tolist())
    #logger.info(f"set_comp_mat: {set_comp_mat}")
    diffnodes = set_dag - set_comp_mat
    logger.debug(f"diffnodes in dag not in comp_mat {diffnodes}")

    # Needs to add final to computation_matrix. It was added to the DAG but not to comp matrix.
    if 'final' in diffnodes:
        d_row = [] # ['final']
        columns = computation_matrix.columns.values.tolist()
        #columns.remove('T')
        for c in columns:
            d_row.append(1)
        logger.info(f"cols {columns} {d_row} {computation_matrix.columns}")
        pd_1 = pd.DataFrame([d_row], columns=computation_matrix.columns, index=['final'])
        #pd_1.set_index('T', inplace=True)
        #pd_1 = pd.DataFrame([['final',1, 1]], columns=computation_matrix.columns)
        logger.info(f"pd_1 {pd_1}")
        computation_matrix = pd.concat([computation_matrix, pd_1]) #, ignore_index=True)

    #print("Concat:\n",computation_matrix)
    #print("final there?", computation_matrix.index[computation_matrix['T'] == 'final'].tolist())
    #print("final there?", computation_matrix.loc['final'])

    # Remove nodes from computation_matrix that are not in the DAG. DAG only contain nodes that have edges.
    # Needs to check why some nodes do not have edges in the dot generated by Legion Spy.
    diffnodes = set_comp_mat - set_dag
    logger.debug(f"diffnodes in comp_mat not in dag {diffnodes}")
    for rem in diffnodes:
        #row = computation_matrix.index[computation_matrix['T'] == rem]
        row = rem #computation_matrix.index[computation_matrix['T'] == rem]
        logger.debug(f"del {rem} row {row}")
        computation_matrix.drop(row, inplace=True)

    set_dag = set(x for i,x in dag.nodes(data="taskinstname"))
    #set_comp_mat =  set(computation_matrix['T'].tolist())
    set_comp_mat =  set(computation_matrix.index.values.tolist())
    diffnodes = set_comp_mat.symmetric_difference(set_dag)
    logger.info(f"Final!!! diffnodes {diffnodes}")

    processor_schedules, _, _ = schedule_dag(dag, communication_matrix=communication_matrix, communication_startup=communication_startup, computation_matrix=computation_matrix, rank_metric=args.rank_metric, mem_mem_matrix=None, mem_pe_matrix=None, mapping=mapping,taskdepweights=taskdepcomm, machinemodel=machmodel, task_inst_names=tasknames, procs=procs, pkindscol=pkindnames)

    totaltime = 0.0
    for proc, jobs in processor_schedules.items():
        logger.debug(f"Processor {proc} has the following jobs:")
        logger.debug(f"\t{jobs}")
        lastjobendtime = jobs[-1].end
        logger.info(f"Processor {proc} has {len(jobs)} jobs and finished at {lastjobendtime}.")
        if lastjobendtime > totaltime:
            totaltime = lastjobendtime

    logger.info(f"Total result time {totaltime:.3f} ns or {totaltime*1.0e-6:.3f} ms.")

    if args.showGantt:
        showGanttChart(processor_schedules)
