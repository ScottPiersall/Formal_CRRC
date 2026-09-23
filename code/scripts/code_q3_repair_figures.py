"""Static audit figures for the supplement; no additional inference."""
import pathlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
O=ROOT/'artifacts/code_extension_v4_q3_continuation';A=O/'analysis'

def main():
    load=lambda p:json.loads(p.read_text())
    curves=load(A/'curve_metrics_by_version.json');outcomes=load(A/'continuation_outcomes.json');changes=load(A/'same_context_scorer_changes.json')
    summaries=load(A/'model_all_valid_summaries.json');status=load(O/'execution_status.json')
    versions=list(curves);names=['Original 8k\noriginal scorer','Original 8k\nshared-prefix scorer','Two-stage 32k\nshared-prefix scorer']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axs[0,0];xx=np.arange(3)
    for offset,v,color in [(-.18,'original','#296caa'),(.18,'explicit','#d56a38')]:
        counts=[sum(r['model_key']=='qwen3' and r['variant']==v and r['representation']=='bare' for r in curves[k]) for k in versions]
        bars=ax.bar(xx+offset,counts,width=.34,label=v,color=color);ax.bar_label(bars,labels=[f'{n}/80' for n in counts],padding=3)
    ax.set(xticks=xx,xticklabels=names,ylim=(0,92),ylabel='Complete main task curves',title='A. Completeness remains explicit');ax.legend(frameon=False)
    ax=axs[0,1];main=[r for r in outcomes if r['split']=='main']
    if main:
        bins=np.linspace(0,24576,13)
        ax.hist([[r['additional_token_count'] for r in main if r['natural_end']],[r['additional_token_count'] for r in main if not r['natural_end']]],
            bins=bins,stacked=True,label=['Natural completion','Terminal failure'],color=['#37887c','#b44949'])
    ax.axvline(24576,color='#222',linestyle='--',linewidth=1)
    ax.set(xlabel='Additional continuation tokens (after original 8192)',ylabel='Main requests',title=f'B. Continuation outcomes: {len(main)}/60 attempted');ax.legend(frameon=False)
    ax=axs[1,0]
    for rep,color in [('bare','#296caa'),('whitespace_union','#d56a38')]:
        vals=[r['margin_delta'] for r in changes if r['representation']==rep and r['split']=='main']
        if vals:ax.hist(vals,bins=40,histtype='step',linewidth=1.5,label=rep,color=color)
    ax.set(xlabel='New margin minus original margin (same reasoning)',ylabel='Main contexts',title='C. Uniform scorer sensitivity');ax.legend(frameon=False)
    ax=axs[1,1];models=['qwen','llama','mistral','gemma','qwen3'];s=summaries[versions[-1]];xx=np.arange(5)
    for offset,v,color in [(-.18,'original','#296caa'),(.18,'explicit','#d56a38')]:
        rows=[s[f'{m}/{v}/bare']['partial'] for m in models]
        y=[r['fsrr']['estimate'] if r['fsrr']['estimate'] is not None else np.nan for r in rows]
        lo=[y[i]-r['fsrr']['ci95'][0] if r['fsrr']['ci95'] else 0 for i,r in enumerate(rows)];hi=[r['fsrr']['ci95'][1]-y[i] if r['fsrr']['ci95'] else 0 for i,r in enumerate(rows)]
        ax.bar(xx+offset,y,width=.34,color=color,label=v,yerr=[lo,hi],capsize=2)
    labels=[f"{label}\n(n={s[f'{m}/original/bare']['partial']['n_tasks']}, {s[f'{m}/explicit/bare']['partial']['n_tasks']})" for m,label in zip(models,['Qwen2.5','Llama3.1','Mistral0.3','Gemma2','Qwen3'])]
    ax.set(xticks=xx,xticklabels=labels,ylim=(0,1.15),ylabel='FSRR',title='D. Partially correct tasks; bare, all-valid sets',xlabel='Valid tasks: original, explicit (out of 23 each)')
    ax.legend(frameon=False)
    fig.suptitle('Post-observation supplement: four immediate readouts + one native reasoning model',fontsize=13)
    for ext in ('png','svg'):fig.savefig(A/f'qwen3_repair_audit.{ext}',dpi=180)
    plt.close(fig)
    (A/'FIGURE_NOTE.md').write_text('PanelD uses each model/template all-valid set, not a silent intersection. Labels show valid task denominators out of23; error bars are the unchanged task-bootstrap95% intervals. Full common-complete and paired estimates are supplied separately. PanelC compares all saved original natural main contexts, including the preserved former mass-validation failure as a numerical diagnostic. No additional inference was used for this figure.\n')
    print('FIGURES_SAVED',A)
if __name__=='__main__':main()
