"""Static research figure from audited saved summaries; no inference."""
import pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_extension as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

o=ROOT/'artifacts/code_extension_v3_five_models/analysis'
summaries=cp.load(o/'model_all_valid_summaries.json')
models=['qwen','llama','mistral','gemma','qwen3']
names=['Qwen2.5-14B','Llama-3.1-8B','Mistral-7B','Gemma-2-9B','Qwen3-30B-A3B (native R-to-S)']
fig,axes=plt.subplots(1,2,figsize=(12,5.3),sharey=True)
for ax,variant in zip(axes,['original','explicit']):
    for j,model in enumerate(models):
        for rep,offset,color,label in [('bare',-.12,'#2563eb','Bare label'),('whitespace_union',.12,'#d97706','Three-form union')]:
            result=summaries[f'{model}/{variant}/{rep}']['partial'];metric=result['fsrr']
            if metric['estimate'] is not None:
                x=metric['estimate'];lo,hi=metric['ci95']
                ax.errorbar(x,j+offset,xerr=[[x-lo],[hi-x]],fmt='o',color=color,capsize=3,markersize=5,label=label if j==0 else None)
        n=summaries[f'{model}/{variant}/bare']['partial']['n_tasks']
        ax.text(1.07,j,f'{n}/23',va='center',ha='center',fontsize=9,color='#475569')
    ax.axhline(3.5,color='#cbd5e1',linewidth=1)
    ax.set(title=variant.capitalize()+' template',xlabel='FSRR; task-bootstrap 95% interval',xlim=(-.03,1.13),yticks=range(5),yticklabels=names)
    ax.set_xticks([0,.25,.5,.75,1]);ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='x',color='#e2e8f0',linewidth=.8);ax.set_axisbelow(True)
axes[0].set_ylim(4.5,-.5)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.64,.93),ncol=2,frameon=False,fontsize=10)
fig.suptitle('Code acceptance recoverability: partially correct candidates (z = 1,...,7)',fontsize=13,y=.98)
fig.text(.5,.015,'Four immediate-readout models; Q3 uses native reason-then-score. Counts show valid tasks / 23. No causal reasoning claim.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.88))
fig.savefig(o/'five_model_recoverability.png',dpi=200);fig.savefig(o/'five_model_recoverability.svg');plt.close(fig)
print('FIGURE_SAVED')
