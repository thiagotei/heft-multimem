"""
Basic implementation of Gantt chart plotting using Matplotlib
Taken from https://sukhbinder.wordpress.com/2016/05/10/quick-gantt-chart-with-matplotlib/ and adapted as necessary (i.e. removed Date logic, etc)
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.colors as mcolors
import numpy as np
import itertools

def ganttChart(proc_schedules, machine=None, title=None):
    """
        Given a dictionary of processor-task schedules, displays a Gantt chart generated using Matplotlib
    """

    processors = list(proc_schedules.keys())
    proclabels = processors
    if machine is not None:
        proclabels = []
        for p in processors:
            proclabels.append(machine.procs[p].kind.value[:3]+str(p))
        ##
    #

    #color_choices = ['red', 'blue', 'green', 'cyan', 'magenta']
    col = itertools.cycle(mcolors.TABLEAU_COLORS)
    task_colors = {}

    ilen=len(processors)
    pos = np.arange(0.5,ilen*0.5+0.5,0.5)
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(111)
    for idx, proc in enumerate(processors):
        for job in proc_schedules[proc]:
            al = {}
            if job.task.name not in task_colors:
                task_colors[job.task.name] = next(col)
                print(f"GANNT task {job.task.name} {task_colors[job.task.name]}")
                al = {'label' : job.task.name}
            #
            c = task_colors[job.task.name]
            ax.barh((idx*0.5)+0.5, job.end - job.start, left=job.start, height=0.3, align='center', edgecolor='black', color=c, alpha=0.95, **al)
            #ax.text(0.5 * (job.start + job.end - len(str(job.task))-0.25), (idx*0.5)+0.5 - 0.03125, job.task+1, color=color_choices[((job.task) // 10) % 5], fontweight='bold', fontsize=18, alpha=0.75)

    locsy, labelsy = plt.yticks(pos, proclabels)
    plt.ylabel('Processor', fontsize=16)
    plt.xlabel('Time', fontsize=16)
    plt.setp(labelsy, fontsize = 14)
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left', borderaxespad=0)
    if title is not None:
        plt.title(title)
    ax.set_ylim(ymin = -0.1, ymax = ilen*0.5+0.5)
    ax.set_xlim(xmin = -5)
    ax.grid(color = 'g', linestyle = ':', alpha=0.5)

    font = font_manager.FontProperties(size='small')

def saveGanttChart(proc_schedules, machine, fname, title=None):
    ganttChart(proc_schedules, machine, title)
    plt.savefig(fname)

def showGanttChart(proc_schedules):
    ganttChart(proc_schedules)
    plt.show()
